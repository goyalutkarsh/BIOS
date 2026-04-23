"""
train_bioanchor_attention_OLD.py
=================================
Self-attention bio head using the OLD architecture:
- network_OLD.py for encoder + projectors (unchanged)
- bio_head is a SEPARATE object with its own bio_optimizer
- Only the bio head uses self-attention instead of linear

This is the old separate-optimizer pattern but with attention
instead of linear, to isolate whether attention itself helps
independent of the shared/separate optimizer question.

Compare results against:
    train_bioanchor_linear_OLD.py  → V=0.4855  (separate optimizer, linear)
    train_bioanchor.py attention   → V=0.4447  (shared optimizer, attention)
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from modules import network_OLD as network, contrastive_loss
from utils import yaml_config_hook
from torch import optim
from dataloader import get_feature
import matplotlib.pyplot as plt


# ── Self-attention bio head (defined inline, no changes to network_OLD.py) ──

class BioAnchorHeadAttention(nn.Module):
    """
    Self-attention bio anchor head.
    Treats each bio dim as its own token — attention learns
    which anchors are relevant to predicting each other.

    Used here with the OLD separate-optimizer architecture
    to isolate attention's effect independent of optimizer choice.
    """
    def __init__(self, bio_dim, d_model=32, num_heads=2):
        super().__init__()
        self.token_proj  = nn.Linear(1, d_model)
        self.attn        = nn.MultiheadAttention(
            embed_dim   = d_model,
            num_heads   = num_heads,
            dropout     = 0.0,
            batch_first = True,
        )
        self.norm        = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, z_bio):
        # (batch, N) → (batch, N, d_model) → (batch, N) 
        x = self.token_proj(z_bio.unsqueeze(-1))
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        return self.output_proj(x).squeeze(-1)


# ── Utilities (same as old train scripts) ──

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
    plt.title('OLD arch + Self-Attention — Train loss vs. epoch')
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{cancer_type}_attention_old_loss.png')
    plt.close()


def save_model(args, model, optimizer, current_epoch):
    os.makedirs(args.model_path, exist_ok=True)
    out = os.path.join(args.model_path, f"checkpoint_{current_epoch}.tar")
    torch.save({
        'net':       model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch':     current_epoch,
    }, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    parser.add_argument("--cancer_type",     "-c", type=str,   required=True)
    parser.add_argument("--batch_size",            type=int,   default=64)
    parser.add_argument("--cluster_number",        type=int,   required=True)
    parser.add_argument("--lambda_bio",            type=float, default=0.1)
    parser.add_argument("--bio_dim",               type=int,   required=True)
    parser.add_argument("--bio_anchor_file",       type=str,   required=True)
    args = parser.parse_args()

    os.makedirs(args.model_path, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"OLD architecture + Self-Attention Bio Head")
    print(f"Cancer: {args.cancer_type} | bio_dim: {args.bio_dim} | lambda_bio: {args.lambda_bio}")
    print(f"Bio anchor file: {args.bio_anchor_file}")

    DL = get_feature(args.cancer_type, args.batch_size, True,
                     bio_anchor_file=args.bio_anchor_file)

    cluster_number = args.cluster_number
    print(f"Cluster number: {cluster_number}")

    from modules.ae import AE
    ae    = AE(hid_dim=args.feature_dim, bio_dim=args.bio_dim)
    model = network.Network(ae, args.feature_dim, cluster_number)

    # bio_head is SEPARATE — old architecture pattern
    bio_head = BioAnchorHeadAttention(bio_dim=args.bio_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model    = model.to(device)
    bio_head = bio_head.to(device)

    # TWO separate optimizers — old architecture pattern
    optimizer     = optim.Adam(model.parameters(),     lr=args.learning_rate, weight_decay=args.weight_decay)
    bio_optimizer = optim.Adam(bio_head.parameters(),  lr=args.learning_rate)

    if args.reload:
        model_fp   = os.path.join(args.model_path, f"checkpoint_{args.start_epoch}.tar")
        checkpoint = torch.load(model_fp)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1

    # loss criteria created once outside loop
    criterion_instance = contrastive_loss.DCL(
        temperature = args.instance_temperature, weight_fn=None)
    criterion_cluster  = contrastive_loss.ClusterLoss(
        cluster_number, args.cluster_temperature, device).to(device)

    loss_epoch_list = []

    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        loss_epoch = 0

        for step, batch_data in enumerate(DL):
            if len(batch_data) == 2:
                x, bio_anchors = batch_data
                x           = x.float().to(device)
                bio_anchors = bio_anchors.float().to(device)
            else:
                x           = batch_data[0].float().to(device)
                bio_anchors = None

            optimizer.zero_grad()
            bio_optimizer.zero_grad()

            x_i = (x + torch.normal(0, 1, size=x.shape)).float().to(device)
            x_j = (x + torch.normal(0, 1, size=x.shape)).float().to(device)

            # get split embeddings
            z_i, z_bio_i, z_novel_i = model.ae(x_i)
            z_j, z_bio_j, z_novel_j = model.ae(x_j)

            # contrastive losses — uses full embedding via model.forward
            z_i_proj, z_j_proj, c_i, c_j = model(x_i, x_j)
            loss_instance = criterion_instance(z_i_proj, z_j_proj)
            loss_cluster  = criterion_cluster(c_i, c_j)

            # bio loss — separate bio_head, same pattern as old linear script
            loss_bio = torch.tensor(0.0).to(device)
            if bio_anchors is not None:
                pred_i   = bio_head(z_bio_i)
                pred_j   = bio_head(z_bio_j)
                loss_bio = (F.mse_loss(pred_i, bio_anchors) +
                            F.mse_loss(pred_j, bio_anchors)) / 2

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
    print("OLD arch + attention training complete.")
