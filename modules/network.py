"""
network.py  —  BIOS model architecture
=======================================
Contains all three bio-anchor head variants in one place.

Head variants (select via --head_type argument):
    'linear'    Scenario 1 — single linear layer         (most interpretable)
    'mlp'       Scenario 2 — two-layer MLP               (balanced)
    'attention' Scenario 3 — multi-head self-attention   (most flexible)

What changed from the old network.py
--------------------------------------
    OLD: BioAnchorHead was a separate class with only a single nn.Linear
         BioAnchorHeadMLP was defined inside train_bioanchor_mlp.py (wrong place)
         Network ignored z_bio and z_novel returned by the ae
         Network.forward returned 4 values: z_i, z_j, c_i, c_j
         Network.forward_cluster returned 2 values: c, h

    NEW: All three head variants live here in one BioAnchorHead class
         Network accepts bio_dim, n_anchors, head_type in __init__
         Network.forward returns 5 values: z_i, z_j, c_i, c_j, b_hat
         Network.forward_cluster returns 3 values: c, h, z_bio
"""

import math
import torch
import torch.nn as nn
from torch.nn.functional import normalize


# ══════════════════════════════════════════════════════════════════
#  Bio Anchor Head
#  Takes z_bio (the bio partition of the embedding) and predicts
#  bio-anchor values b_hat.
#
#  OLD code (was its own class, linear only):
#      class BioAnchorHead(nn.Module):
#          def __init__(self, bio_dim=15, n_anchors=15):
#              self.predictor = nn.Linear(bio_dim, n_anchors)   # <-- old, linear only
#          def forward(self, z_bio):
#              return self.predictor(z_bio)
#
#  NEW: same class name, now supports all three variants via head_type
# ══════════════════════════════════════════════════════════════════

class BioAnchorHead(nn.Module):

    def __init__(self, bio_dim, n_anchors, head_type):
        """
        Args:
            bio_dim   : number of bio dims in the split embedding (e.g. 5)
            n_anchors : number of bio anchors to predict (usually == bio_dim)
            head_type : 'linear' | 'mlp' | 'attention'
        """
        super(BioAnchorHead, self).__init__()

        self.head_type = head_type

        # ── Scenario 1: Linear ──────────────────────────────────
        # Single linear layer. Equivalent to the old BioAnchorHead.
        # Most interpretable — weight matrix directly shows which
        # bio dims contribute to each anchor prediction.
        # z_bio (batch, N) → Linear → b_hat (batch, N)
        if head_type == 'linear':
            self.predictor = nn.Linear(bio_dim, n_anchors)

        # ── Scenario 2: MLP ─────────────────────────────────────
        # Two-layer MLP with ReLU. Was previously defined inline
        # inside train_bioanchor_mlp.py — moved here where it belongs.
        # Can learn non-linear interactions between bio dims
        # e.g. how proliferation and immune scores interact.
        # z_bio (batch, N) → Linear(N,64) → ReLU → Linear(64,N) → b_hat
        elif head_type == 'mlp':
            self.predictor = nn.Sequential(
                nn.Linear(bio_dim, 64),
                nn.ReLU(),
                nn.Linear(64, n_anchors),
            )

        # ── Scenario 3: Self-Attention ───────────────────────────
        # Each bio dim is treated as its own token. Attention lets
        # the model learn which bio-anchors are relevant to each
        # other — e.g. Apoptosis <-> Proliferation inverse relation,
        # Hypoxia and EMT co-activating in aggressive subtypes.
        # The learned N×N attention weight matrix is also a free
        # interpretability signal (which anchor attends to which).
        #
        # Flow:
        #   z_bio (batch, N)
        #     unsqueeze   → (batch, N, 1)
        #     token_proj  → (batch, N, d_model)  lift each scalar to d_model dims
        #     self-attn   → (batch, N, d_model)  each token attends to all others
        #     Add & LN    → (batch, N, d_model)  residual + layer norm
        #     output_proj → (batch, N, 1)         collapse back to scalar
        #     squeeze     → (batch, N)            = b_hat
        elif head_type == 'attention':
            d_model   = 32   # token embedding dim; 32 is enough for N=5-8
            num_heads = 2    # 2 attention heads, each with d_k=16

            # lifts each scalar bio dim to a d_model-dimensional token
            self.token_proj = nn.Linear(1, d_model)

            # PyTorch built-in multi-head attention
            # batch_first=True: input shape is (batch, seq_len, dim)
            self.attn = nn.MultiheadAttention(
                embed_dim   = d_model,
                num_heads   = num_heads,
                dropout     = 0.0,
                batch_first = True,
            )

            # layer norm for the residual connection
            self.norm = nn.LayerNorm(d_model)

            # collapses each d_model-dim token back to a single scalar
            self.output_proj = nn.Linear(d_model, 1)

        else:
            raise ValueError(
                f"Unknown head_type '{head_type}'. "
                f"Choose: 'linear', 'mlp', or 'attention'"
            )

    def forward(self, z_bio):
        """
        Args:
            z_bio : (batch, bio_dim)   — bio partition of the embedding
        Returns:
            b_hat : (batch, n_anchors) — predicted bio-anchor values
        """

        if self.head_type in ('linear', 'mlp'):
            # straight through — same as old BioAnchorHead.forward()
            return self.predictor(z_bio)

        elif self.head_type == 'attention':
            # treat each bio dim as an independent token
            x = z_bio.unsqueeze(-1)            # (batch, N, 1)
            x = self.token_proj(x)             # (batch, N, d_model)

            # self-attention: every token looks at every other token
            # attn_out contains contextualised token representations
            attn_out, _ = self.attn(x, x, x)  # (batch, N, d_model)

            # residual connection + layer norm (stabilises gradients)
            x = self.norm(x + attn_out)        # (batch, N, d_model)

            # collapse each token back to a scalar prediction
            b_hat = self.output_proj(x)        # (batch, N, 1)
            return b_hat.squeeze(-1)           # (batch, N)


# ══════════════════════════════════════════════════════════════════
#  Network
#  Wraps the autoencoder and adds three heads on top:
#    1. instance_projector  → for instance-level contrastive loss
#    2. cluster_projector   → for cluster-level contrastive loss
#    3. bio_head            → for bio-anchor prediction (NEW)
#
#  OLD __init__ signature:
#      def __init__(self, ae, feature_dim, class_num):
#
#  NEW __init__ signature:
#      def __init__(self, ae, feature_dim, class_num,
#                   bio_dim, n_anchors, head_type):
# ══════════════════════════════════════════════════════════════════

class Network(nn.Module):

    def __init__(self, ae, feature_dim, class_num,
                 bio_dim, n_anchors, head_type):
        """
        Args:
            ae          : autoencoder module (from modules/ae.py)
            feature_dim : output dim of instance projector
            class_num   : number of clusters
            bio_dim     : number of bio dims in the embedding split  # NEW
            n_anchors   : number of bio anchors to predict           # NEW
            head_type   : 'linear' | 'mlp' | 'attention'            # NEW
        """
        super(Network, self).__init__()

        self.ae          = ae
        self.feature_dim = feature_dim
        self.cluster_num = class_num

        # ── instance contrastive projector (unchanged from old code) ──
        # maps full embedding h → normalised z for DCL loss
        self.instance_projector = nn.Sequential(
            nn.Linear(self.ae.rep_dim, self.ae.rep_dim),
            nn.ReLU(),
            nn.Linear(self.ae.rep_dim, self.feature_dim),
        )

        # ── cluster projector (unchanged from old code) ──
        # maps full embedding h → soft cluster assignment c
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.ae.rep_dim, self.ae.rep_dim),
            nn.ReLU(),
            nn.Linear(self.ae.rep_dim, self.cluster_num),
            nn.Softmax(dim=1)
        )

        # ── bio anchor head (NEW) ──
        # maps z_bio → predicted bio-anchor values b_hat
        # head_type selects which of the three variants to use
        self.bio_head = BioAnchorHead(bio_dim, n_anchors, head_type)

    def forward(self, x_i, x_j):
        """
        Forward pass for training.
        Runs both augmented views through the ae and all three heads.

        Args:
            x_i, x_j : augmented views of same batch, shape (batch, input_dim)

        Returns:
            z_i, z_j : instance projector outputs for contrastive loss
            c_i, c_j : cluster projector outputs for cluster loss
            b_hat    : bio-anchor predictions from view i   (NEW)

        OLD returned: z_i, z_j, c_i, c_j          (4 values)
        NEW returns:  z_i, z_j, c_i, c_j, b_hat   (5 values)
        Update your training script to unpack 5 values.
        """

        # OLD: h_i, _, _ = self.ae(x_i)  ← was discarding z_bio, z_novel
        # NEW: unpack all three so we can use z_bio for the bio head
        h_i, z_bio_i, z_novel_i = self.ae(x_i)
        h_j, z_bio_j, z_novel_j = self.ae(x_j)

        # instance contrastive head — unchanged
        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_j = normalize(self.instance_projector(h_j), dim=1)

        # cluster head — unchanged
        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)

        # bio anchor prediction — NEW
        # only run on view i here; training script handles both views
        # by calling bio_head on z_bio_i and z_bio_j separately
        b_hat = self.bio_head(z_bio_i)

        return z_i, z_j, c_i, c_j, b_hat

    def forward_cluster(self, x):
        """
        Inference pass — returns cluster assignments and embeddings.
        Used during evaluation (no gradients needed).

        OLD returned: c, h          (2 values)
        NEW returns:  c, h, z_bio   (3 values)
        Update your inference() function to unpack 3 values.
        """

        # OLD: h, _, _ = self.ae(x)  ← was discarding z_bio
        h, z_bio, z_novel = self.ae(x)

        c = self.cluster_projector(h)
        c = torch.argmax(c, dim=1)

        # z_bio returned so evaluation can inspect bio predictions
        return c, h, z_bio   # OLD returned: c, h