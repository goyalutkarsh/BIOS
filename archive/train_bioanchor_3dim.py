import os
import numpy as np
import torch
import argparse
from modules import network, contrastive_loss
from modules.network import BioAnchorHead
from utils import yaml_config_hook
from torch import optim
from dataloader import get_feature
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.cluster import KMeans
import copy
import matplotlib.pyplot as plt
import torch.nn.functional as F


def inference(loader, model, device):
    model.eval()
    cluster_vector = []
    feature_vector = []
    for step, batch_data in enumerate(loader):
        if len(batch_data) == 2:
            x, _ = batch_data
        else:
            x = batch_data[0]
        x = x.float().to(device)
        with torch.no_grad():
            z, _, _ = model.ae(x)
            c, h = model.forward_cluster(x)
        c = c.detach()
        h = h.detach()
        cluster_vector.extend(c.cpu().detach().numpy())
        feature_vector.extend(h.cpu().detach().numpy())
    cluster_vector = np.array(cluster_vector)
    feature_vector = np.array(feature_vector)
    print("Features shape {}".format(feature_vector.shape))
    return cluster_vector, feature_vector


def draw_fig(loss, cancer_type, epoch):
    x = np.array(range(0, epoch + 1))
    y = np.array(loss)
    plt.plot(x, y, marker='o')
    plt.xlabel('epoch')
    plt.ylabel('Train loss')
    plt.title('Train loss vs. epoch')
    save_file = 'results/' + cancer_type + 'Train_loss.png'
    plt.savefig(save_file)

def save_model(args, model, optimizer, current_epoch):
    out = os.path.join(args.model_path, "checkpoint_{}.tar".format(current_epoch))
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': current_epoch}
    torch.save(state, out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    parser.add_argument("--cancer_type", "-c", type=str)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--cluster_number", type=int)
    parser.add_argument("--lambda_bio", type=float, default=0.5, help="Bio-anchor loss weight")
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # prepare data
    DL = get_feature(args.cancer_type, args.batch_size, True)

    print(args.cluster_number)
    cluster_number = args.cluster_number

    # initialize model
    from modules.ae_3dim import AE
    ae = AE(hid_dim=args.feature_dim, bio_dim=3)
    model = network.Network(ae, args.feature_dim, cluster_number)
    
    # Initialize bio-anchor head with 3 dimensions
    bio_head = BioAnchorHead(bio_dim=3, n_anchors=3)
    
    # setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    bio_head = bio_head.to(device)
    
    # optimizer / loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    bio_optimizer = optim.Adam(bio_head.parameters(), lr=args.learning_rate)
    
    if args.reload:
        model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.start_epoch))
        checkpoint = torch.load(model_fp)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1
    
    loss_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train
    loss_epoch_list = []

    for epoch in range(args.start_epoch, args.epochs):
        loss_epoch = 0
        for step, batch_data in enumerate(DL):
            # Handle data with or without bio-anchors
            if len(batch_data) == 2:
                x, bio_anchors = batch_data
                x = x.float().to(device)
                bio_anchors = bio_anchors.float().to(device)
            else:
                x = batch_data[0].float().to(device)
                bio_anchors = None
            
            optimizer.zero_grad()
            if bio_anchors is not None:
                bio_optimizer.zero_grad()
            
            # Data augmentation
            x_i = (x + torch.normal(0, 1, size=(x.shape[0], x.shape[1]))).float().to(device)
            x_j = (x + torch.normal(0, 1, size=(x.shape[0], x.shape[1]))).float().to(device)
            
            # Forward pass - get split embeddings
            z_i, z_bio_i, z_novel_i = model.ae(x_i)
            z_j, z_bio_j, z_novel_j = model.ae(x_j)
            
            # Contrastive learning (uses full embeddings)
            z_i_proj, z_j_proj, c_i, c_j = model(x_i, x_j)
            
            # Instance-level contrastive loss
            criterion_instance = contrastive_loss.DCL(temperature=args.instance_temperature, weight_fn=None)
            loss_instance = criterion_instance(z_i_proj, z_j_proj)
            
            # Cluster-level contrastive loss
            criterion_cluster = contrastive_loss.ClusterLoss(cluster_number, args.cluster_temperature, loss_device).to(loss_device)
            loss_cluster = criterion_cluster(c_i, c_j)
            
            # Bio-anchor prediction loss (weak supervision)
            loss_bio = torch.tensor(0.0).to(device)
            if bio_anchors is not None:
                # For 3D: select 2 out of 3 anchors (67% weak supervision)
                num_bio_anchors = bio_anchors.shape[1]  # Will be 3
                n_selected = 2  # Select 2 out of 3
                selected_indices = np.random.choice(num_bio_anchors, n_selected, replace=False)
                
                # Predict bio-anchors from z_bio
                pred_anchors_i = bio_head(z_bio_i)
                pred_anchors_j = bio_head(z_bio_j)
                
                # Compute MSE loss only on selected anchors
                loss_bio = (
                    F.mse_loss(pred_anchors_i[:, selected_indices], bio_anchors[:, selected_indices]) +
                    F.mse_loss(pred_anchors_j[:, selected_indices], bio_anchors[:, selected_indices])
                ) / 2
            
            # Total loss
            loss = loss_instance + loss_cluster + args.lambda_bio * loss_bio
            
            loss.backward()
            optimizer.step()
            if bio_anchors is not None:
                bio_optimizer.step()
            
            if step % 50 == 0:
                print(f"Epoch [{epoch}/{args.epochs}]\t Step [{step}/{len(DL)}]\t Loss: {loss.item():.4f}\t Bio Loss: {loss_bio.item():.4f}")
            
            loss_epoch += loss.item()
        
        loss_epoch_list.append(loss_epoch)
        print(f"Epoch [{epoch}/{args.epochs}] Loss: {loss_epoch}")
        
        if (epoch % 10 == 0) or (epoch == args.epochs - 1):
            save_model(args, model, optimizer, epoch)

    print(range(args.start_epoch, args.epochs))
    draw_fig(loss_epoch_list, args.cancer_type, args.epochs - 1)