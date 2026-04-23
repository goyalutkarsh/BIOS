import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


def get_feature(cancer_type, batch_size, training, bio_anchor_file=None):
    """
    Load multi-omics features and optionally bio-anchors.

    Parameters
    ----------
    cancer_type    : str   e.g. "BRCA", "LUAD"
    batch_size     : int
    training       : bool  shuffle if True
    bio_anchor_file: str or None
                     Path to bio-anchors CSV (e.g. "bio_anchors_BRCA_6dim.csv").
                     If None, no bio-anchors are loaded and dataloader returns
                     only omics features.
    """
    # ── Load omics features ───────────────────────────────────────────────────
    base = f'../subtype_file/fea/{cancer_type}'
    fea_CN    = pd.read_csv(f'{base}/CN.fea',    header=0, index_col=0, sep=',')
    fea_meth  = pd.read_csv(f'{base}/meth.fea',  header=0, index_col=0, sep=',')
    fea_mirna = pd.read_csv(f'{base}/miRNA.fea', header=0, index_col=0, sep=',')
    fea_rna   = pd.read_csv(f'{base}/rna.fea',   header=0, index_col=0, sep=',')

    feature = np.concatenate((fea_CN, fea_meth, fea_mirna, fea_rna), axis=0).T
    feature = MinMaxScaler().fit_transform(feature)
    feature = torch.tensor(feature, dtype=torch.float32)

    # ── Load bio-anchors (optional) ───────────────────────────────────────────
    anchor_values = None
    if bio_anchor_file is not None:
        if not os.path.exists(bio_anchor_file):
            print(f"[dataloader] Warning: bio_anchor_file not found: {bio_anchor_file}")
        else:
            bio_anchors = pd.read_csv(bio_anchor_file)
            patient_ids = fea_CN.columns.tolist()
            bio_anchors = bio_anchors.set_index('patient_id').loc[patient_ids].reset_index()
            anchor_values = torch.tensor(bio_anchors.iloc[:, 1:].values, dtype=torch.float32)
            print(f"[dataloader] Loaded bio-anchors from {bio_anchor_file}: {anchor_values.shape}")
    else:
        print(f"[dataloader] No bio_anchor_file specified — loading omics only")

    # ── Build dataset and dataloader ──────────────────────────────────────────
    dataset = TensorDataset(feature, anchor_values) if anchor_values is not None else TensorDataset(feature)
    return DataLoader(dataset, batch_size=batch_size, shuffle=training)