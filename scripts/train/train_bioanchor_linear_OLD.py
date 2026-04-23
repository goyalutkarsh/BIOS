import os
import numpy as np
import torch
import argparse
from modules import network_OLD as network, contrastive_loss
from modules.network_OLD import BioAnchorHead
from utils import yaml_config_hook
from torch import optim
from dataloader import get_feature
import matplotlib.pyplot as plt
import torch.nn.functional as F


def inference(loader, model, device):
    model.eval()
    cluster_vector, feature_vector = [], []
    for step, batch_data in enumerate(loader):
        if len(batch_data) == 2:
            x, _ = batch_data
        else:
            x = batch_data[0]
        x = x.float().to(device)
        with torch.no_grad():
            z, _, _ = model.ae(x)
            c, h = model.forward_cluster(x)
        cluster_vector.extend(c.cpu().detach().numpy())
        feature_vector.extend(h.cpu().detach().numpy())
    print("Features shape {}".format(np.array(feature_vector).shape))
    return np.array(cluster_vector), np.array(feature_vector)


def draw_fig(loss, cancer_type, epoch):
    plt.figure()
    plt.plot(range(len(loss)), loss, marker='o')
    plt.xlabel('epoch')
    plt.ylabel('Train loss')
    plt.title('Scenario 1 (Direct Linear) - Train loss vs. epoch')
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{cancer_type}_linear_loss.png')
    plt.close()


def save_model(args, model, optimizer, current_epoch):
    os.makedirs(args.model_path, exist_ok=True)
    out = os.path.join(args.model_path, f"checkpoint_{current_epoch}.tar")
    torch.save({'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': current_epoch}, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    parser.add_argument("--cancer_type", "-c", type=str)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--cluster_number", type=int)
    parser.add_argument("--lambda_bio", type=float, default=0.1)
    parser.add_argument("--bio_dim", type=int, default=6,
                        help="Number of bio-anchor dimensions (e.g. 3, 6, 10)")
    parser.add_argument("--bio_anchor_file", type=str, default=None,
                        help="Path to bio-anchors CSV. If not provided, trains without bio supervision.")
    args = parser.parse_args()

    os.makedirs(args.model_path, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Linear Bio-Anchor Head")
    print(f"Cancer: {args.cancer_type} | bio_dim: {args.bio_dim} | lambda_bio: {args.lambda_bio}")
    print(f"Bio anchor file: {args.bio_anchor_file}")

    DL = get_feature(args.cancer_type, args.batch_size, True,
                     bio_anchor_file=args.bio_anchor_file)

    cluster_number = args.cluster_number
    print(f"Cluster number: {cluster_number}")

    from modules.ae import AE
    ae = AE(hid_dim=args.feature_dim, bio_dim=args.bio_dim)
    model = network.Network(ae, args.feature_dim, cluster_number)

    # Scenario 1: direct linear — bio_dim -> bio_dim (single linear layer)
    bio_head = BioAnchorHead(bio_dim=args.bio_dim, n_anchors=args.bio_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    bio_head = bio_head.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    bio_optimizer = optim.Adam(bio_head.parameters(), lr=args.learning_rate)

    if args.reload:
        model_fp = os.path.join(args.model_path, f"checkpoint_{args.start_epoch}.tar")
        checkpoint = torch.load(model_fp)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1

    loss_epoch_list = []

    for epoch in range(args.start_epoch, args.epochs):
        loss_epoch = 0
        for step, batch_data in enumerate(DL):
            if len(batch_data) == 2:
                x, bio_anchors = batch_data
                x = x.float().to(device)
                bio_anchors = bio_anchors.float().to(device)
            else:
                x = batch_data[0].float().to(device)
                bio_anchors = None

            optimizer.zero_grad()
            bio_optimizer.zero_grad()

            x_i = (x + torch.normal(0, 1, size=x.shape)).float().to(device)
            x_j = (x + torch.normal(0, 1, size=x.shape)).float().to(device)

            z_i, z_bio_i, z_novel_i = model.ae(x_i)
            z_j, z_bio_j, z_novel_j = model.ae(x_j)
            z_i_proj, z_j_proj, c_i, c_j = model(x_i, x_j)

            criterion_instance = contrastive_loss.DCL(temperature=args.instance_temperature, weight_fn=None)
            loss_instance = criterion_instance(z_i_proj, z_j_proj)

            criterion_cluster = contrastive_loss.ClusterLoss(cluster_number, args.cluster_temperature, device).to(device)
            loss_cluster = criterion_cluster(c_i, c_j)

            loss_bio = torch.tensor(0.0).to(device)
            if bio_anchors is not None:
                pred_i = bio_head(z_bio_i)
                pred_j = bio_head(z_bio_j)
                loss_bio = (F.mse_loss(pred_i, bio_anchors) + F.mse_loss(pred_j, bio_anchors)) / 2

            loss = loss_instance + loss_cluster + args.lambda_bio * loss_bio
            loss.backward()
            optimizer.step()
            bio_optimizer.step()

            if step % 50 == 0:
                print(f"Epoch [{epoch}/{args.epochs}] Step [{step}/{len(DL)}] "
                      f"Loss: {loss.item():.4f} Bio: {loss_bio.item():.4f}")

            loss_epoch += loss.item()

        loss_epoch_list.append(loss_epoch)
        print(f"Epoch [{epoch}/{args.epochs}] Total Loss: {loss_epoch:.4f}")

        if epoch % 10 == 0:
            save_model(args, model, optimizer, epoch)

    save_model(args, model, optimizer, args.epochs)
    draw_fig(loss_epoch_list, args.cancer_type, args.epochs - 1)
    print("Linear training complete.")
