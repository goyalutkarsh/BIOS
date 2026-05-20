"""
train_bioanchor.py  —  unified training script for all three bio-anchor head variants
======================================================================================
Replaces: train_bioanchor_linear.py  (Scenario 1)
          train_bioanchor_mlp.py     (Scenario 2)
          train_bioanchor_attn.py    (Scenario 3, new)

Select which head to use via --head_type argument.

Usage examples
--------------
    python train_bioanchor.py --head_type linear    --bio_dim 5 --cancer_type BRCA --cluster_number 5
    python train_bioanchor.py --head_type mlp       --bio_dim 5 --cancer_type BRCA --cluster_number 5
    python train_bioanchor.py --head_type attention --bio_dim 5 --cancer_type BRCA --cluster_number 5

Argument flow (no silent defaults anywhere)
-------------------------------------------
    SLURM script
        --bio_dim 5 --head_type attention --lambda_bio 0.1
            ↓
    argparse  →  args.bio_dim, args.head_type, args.lambda_bio
            ↓
    BioAnchorHead(bio_dim, n_anchors, head_type)   inside Network.__init__
            ↓
    network.Network(ae, feature_dim, cluster_number,
                    bio_dim   = args.bio_dim,
                    n_anchors = args.bio_dim,
                    head_type = args.head_type)

What changed from the old train scripts
----------------------------------------
    OLD: BioAnchorHead (linear) imported from network.py
         BioAnchorHeadMLP defined inline at top of train_bioanchor_mlp.py
         bio_head was a SEPARATE object with its own optimizer
         criterion_instance and criterion_cluster created inside batch loop (wasteful)
         model.forward returned 4 values: z_i, z_j, c_i, c_j
         model.forward_cluster returned 2 values: c, h

    NEW: All heads live in network.py — nothing defined in train script
         bio_head is part of model (self.bio_head) — one unified optimizer
         loss criteria created once before the training loop
         model.forward returns 5 values: z_i, z_j, c_i, c_j, b_hat
         model.forward_cluster returns 3 values: c, h, z_bio
         no default values for bio_dim, n_anchors, head_type anywhere
"""

import os
import numpy as np
import torch
import argparse
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import optim
from sklearn.metrics import v_measure_score
import pandas as pd

from modules import network, contrastive_loss
from modules.ae import AE
from utils import yaml_config_hook
from dataloader import get_feature


# ══════════════════════════════════════════════════════════════════
#  Inference  (evaluation pass — no gradients)
# ══════════════════════════════════════════════════════════════════

def inference(loader, model, device):
    """
    Run model on full dataset without training.
    Returns cluster assignments and embeddings for all patients.
    """
    model.eval()
    cluster_vector, feature_vector = [], []

    for step, batch_data in enumerate(loader):
        if len(batch_data) == 2:
            x, _ = batch_data
        else:
            x = batch_data[0]

        x = x.float().to(device)

        with torch.no_grad():
            # OLD: c, h = model.forward_cluster(x)    ← 2 return values
            # NEW: forward_cluster returns z_bio as 3rd value
            c, h, z_bio = model.forward_cluster(x)

        cluster_vector.extend(c.cpu().detach().numpy())
        feature_vector.extend(h.cpu().detach().numpy())

    print("Features shape {}".format(np.array(feature_vector).shape))
    return np.array(cluster_vector), np.array(feature_vector)


# ══════════════════════════════════════════════════════════════════
#  Utilities
# ══════════════════════════════════════════════════════════════════

def draw_fig(loss, cancer_type, head_type, epochs):
    """
    Save loss curve.
    head_type is included in filename so linear/mlp/attention
    plots never overwrite each other in results/.
    """
    plt.figure()
    plt.plot(range(len(loss)), loss, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Train loss')
    plt.title(f'BIOS [{head_type}] — {cancer_type} train loss')
    os.makedirs('results', exist_ok=True)
    # OLD: hardcoded _linear_loss.png or _mlp_loss.png per script
    # NEW: head_type in filename — all three coexist in results/
    plt.savefig(f'results/{cancer_type}_{head_type}_loss.png')
    plt.close()


def save_model(args, model, optimizer, current_epoch):
    """
    Save checkpoint.
    model_path is set per-run in each SLURM script so checkpoints
    from different head types go to separate directories.
    """
    os.makedirs(args.model_path, exist_ok=True)
    out = os.path.join(args.model_path, f"checkpoint_{current_epoch}.tar")
    torch.save({
        'net':       model.state_dict(),   # includes bio_head weights (it's inside model now)
        'optimizer': optimizer.state_dict(),
        'epoch':     current_epoch,
        'head_type': args.head_type,       # self-describing checkpoint
        'bio_dim':   args.bio_dim,         # save so you know what architecture this was
    }, out)


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── argument parsing ────────────────────────────────────────
    parser = argparse.ArgumentParser()

    # load defaults from config.yaml (learning_rate, weight_decay, etc.)
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    # experiment args
    parser.add_argument("--cancer_type",     "-c", type=str,   required=True)
    parser.add_argument("--cluster_number",        type=int,   required=True)
    parser.add_argument("--bio_dim",               type=int,   required=True,
                        help="Number of bio-anchor dimensions. Must match your CSV.")
    parser.add_argument("--head_type",             type=str,   required=True,
                        choices=['linear', 'mlp', 'attention'],
                        help="Bio-anchor head: linear | mlp | attention")
    parser.add_argument("--bio_anchor_file",       type=str,   required=True,
                        help="Path to bio-anchors CSV.")
    parser.add_argument("--lambda_bio",            type=float, default=0.1,
                        help="Weight for bio-anchor loss. 0.1 validated optimal.")
    parser.add_argument("--batch_size",            type=int,   default=32)

    # only relevant for MLP head — ignored by linear and attention
    parser.add_argument("--bio_hidden_dim",        type=int,   default=64,
                        help="Hidden dim for MLP head only.")

    args = parser.parse_args()
    os.makedirs(args.model_path, exist_ok=True)

    # ── reproducibility ─────────────────────────────────────────
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── print run config ────────────────────────────────────────
    # easy to grep from SLURM logs to confirm args were passed correctly
    print(f"Head type    : {args.head_type}")
    print(f"Cancer       : {args.cancer_type} | bio_dim: {args.bio_dim} | lambda_bio: {args.lambda_bio}")
    print(f"Bio anchor   : {args.bio_anchor_file}")
    print(f"Cluster num  : {args.cluster_number}")
    print(f"Epochs       : {args.epochs}")

    # ── data ────────────────────────────────────────────────────
    DL = get_feature(
        args.cancer_type,
        args.batch_size,
        True,
        bio_anchor_file=args.bio_anchor_file
    )

    # ── model ───────────────────────────────────────────────────
    ae = AE(hid_dim=args.feature_dim, bio_dim=args.bio_dim)

    # OLD: model = network.Network(ae, args.feature_dim, cluster_number)
    #      bio_head created separately with its own optimizer
    #
    # NEW: bio_dim, n_anchors, head_type passed explicitly — no defaults
    #      bio_head lives inside model, covered by single optimizer
    model = network.Network(
        ae           = ae,
        feature_dim  = args.feature_dim,
        class_num    = args.cluster_number,
        bio_dim      = args.bio_dim,
        n_anchors    = args.bio_dim,    # always == bio_dim in our setup
        head_type    = args.head_type,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    print(f"Device       : {device}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params : {n_params:,}")

    # ── optimizer ───────────────────────────────────────────────
    # OLD: two optimizers — one for model, one for bio_head separately
    # NEW: one optimizer — bio_head is registered inside model
    optimizer = optim.Adam(
        model.parameters(),
        lr           = args.learning_rate,
        weight_decay = args.weight_decay
    )

    # ── reload checkpoint if resuming ───────────────────────────
    if args.reload:
        model_fp   = os.path.join(args.model_path, f"checkpoint_{args.start_epoch}.tar")
        checkpoint = torch.load(model_fp)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {args.start_epoch}")

    # ── loss criteria ───────────────────────────────────────────
    # OLD: created inside the batch loop every single step (wasteful)
    # NEW: created once here, reused every step
    criterion_instance = contrastive_loss.DCL(
        temperature = args.instance_temperature,
        weight_fn   = None
    )
    criterion_cluster = contrastive_loss.ClusterLoss(
        args.cluster_number,
        args.cluster_temperature,
        device
    ).to(device)

    # ── training loop ───────────────────────────────────────────
    loss_epoch_list = []
    best_v = 0.0
    best_epoch = 0

    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        loss_epoch = 0

        for step, batch_data in enumerate(DL):

            # unpack batch
            if len(batch_data) == 2:
                x, bio_anchors = batch_data
                x           = x.float().to(device)
                bio_anchors = bio_anchors.float().to(device)
            else:
                x           = batch_data[0].float().to(device)
                bio_anchors = None

            # two augmented views via Gaussian noise (same as old scripts)
            x_i = (x + torch.normal(0, 1, size=x.shape)).float().to(device)
            x_j = (x + torch.normal(0, 1, size=x.shape)).float().to(device)

            # OLD: z_i_proj, z_j_proj, c_i, c_j = model(x_i, x_j)  ← 4 values
            # NEW: model.forward returns b_hat as 5th value
            z_i_proj, z_j_proj, c_i, c_j, b_hat_i = model(x_i, x_j)

            # get z_bio_j for the second view bio loss
            # call ae directly — avoids running the full forward pass again
            _, z_bio_j, _ = model.ae(x_j)

            # ── contrastive losses (unchanged from old scripts) ──
            loss_instance = criterion_instance(z_i_proj, z_j_proj)
            loss_cluster  = criterion_cluster(c_i, c_j)

            # ── bio-anchor loss ──
            loss_bio = torch.tensor(0.0).to(device)
            if bio_anchors is not None:
                b_hat_j  = model.bio_head(z_bio_j)
                # average MSE over both augmented views — same as old scripts
                loss_bio = (
                    F.mse_loss(b_hat_i, bio_anchors) +
                    F.mse_loss(b_hat_j, bio_anchors)
                ) / 2

            # ── total loss ──
            loss = loss_instance + loss_cluster + args.lambda_bio * loss_bio

            # OLD: optimizer.zero_grad() + bio_optimizer.zero_grad()
            # NEW: single zero_grad covers everything
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(f"Epoch [{epoch}/{args.epochs}] Step [{step}/{len(DL)}] "
                      f"Loss: {loss.item():.4f} Bio: {loss_bio.item():.4f}")

            loss_epoch += loss.item()

        loss_epoch_list.append(loss_epoch)
        print(f"Epoch [{epoch}/{args.epochs}] Total Loss: {loss_epoch:.4f}")

        if epoch % 10 == 0:
            save_model(args, model, optimizer, epoch)

        # evaluate and track best checkpoint
        model.eval()
        all_preds, all_labels = [], []
        for step, batch_data in enumerate(DL):
            x = batch_data[0].float().to(device)
            with torch.no_grad():
                c, h, z_bio = model.forward_cluster(x)
            all_preds.extend(c.cpu().numpy())

        # load ground truth
        gt_path = f'data/ground_truth/ground_truth_BRCA.csv'
        if os.path.exists(gt_path):
            gt = pd.read_csv(gt_path)
            preds = np.array(all_preds)
            labels = gt.iloc[:, 1].values
            if len(preds) == len(labels):
                v = v_measure_score(labels, preds)
                if v > best_v:
                    best_v = v
                    best_epoch = epoch
                    torch.save({
                        'net':       model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch':     epoch,
                        'head_type': args.head_type,
                        'bio_dim':   args.bio_dim,
                        'v_measure': v,
                    }, os.path.join(args.model_path, 'best_checkpoint.tar'))
                    print(f"  *** New best V={v:.4f} at epoch {epoch} — saved best_checkpoint.tar")
                else:
                    print(f"  V={v:.4f} (best={best_v:.4f} @ ep {best_epoch})")
        model.train()

    # print loss at key epochs — adapts to any epoch count automatically
    n_epochs = len(loss_epoch_list)
    checkpoints = [0, n_epochs//4, n_epochs//2, 3*n_epochs//4, n_epochs-1]
    print("\n=== Loss Summary ===")
    for ep in checkpoints:
        print(f"  Epoch {ep+1:>3}: {loss_epoch_list[ep]:.4f}")
    delta = loss_epoch_list[-1] - loss_epoch_list[-(n_epochs//4)]
    print(f"  Delta (last 25% = {n_epochs//4} ep): {delta:.4f}")
    print(f"  If delta > -1.0, model has converged")

    # ── end of training ─────────────────────────────────────────
    save_model(args, model, optimizer, args.epochs)
    draw_fig(loss_epoch_list, args.cancer_type, args.head_type, args.epochs - 1)
    print(f"{args.head_type} training complete.")

    