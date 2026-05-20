"""
generate_fig_correlation.py
Run on HPC:
    cd /gpfs/research/fangroup/ug25b/BIOS/Subtype-DCC
    PYBIN=/gpfs/home/ug25b/.conda/envs/bios/bin/python
    $PYBIN generate_fig_correlation.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

anchors = pd.read_csv('data/bio_anchors/bio_anchors_BRCA_gsva_M8.csv')

anchor_cols = [c for c in anchors.columns if c != 'patient_id']

short_labels = {
    'HALLMARK_INFLAMMATORY_RESPONSE':             'Inflammatory',
    'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION': 'EMT',
    'HALLMARK_DNA_REPAIR':                        'DNA Repair',
    'HALLMARK_APOPTOSIS':                         'Apoptosis',
    'HALLMARK_E2F_TARGETS':                       'E2F Targets',
    'HALLMARK_MYC_TARGETS_V1':                    'MYC Targets',
    'HALLMARK_PI3K_AKT_MTOR_SIGNALING':           'PI3K/AKT',
    'HALLMARK_ANGIOGENESIS':                      'Angiogenesis',
}

corr = anchors[anchor_cols].corr()
corr.index   = [short_labels[c] for c in corr.index]
corr.columns = [short_labels[c] for c in corr.columns]

desired_order = [
    'Inflammatory', 'Apoptosis', 'Angiogenesis', 'EMT',
    'DNA Repair', 'E2F Targets', 'MYC Targets', 'PI3K/AKT'
]
corr = corr.loc[desired_order, desired_order]

fig, ax = plt.subplots(figsize=(10, 8.5))

sns.heatmap(
    corr,
    ax=ax,
    cmap='RdBu_r',
    vmin=-1, vmax=1,
    annot=True,
    fmt='.2f',
    annot_kws={'size': 12},
    linewidths=0.5,
    linecolor='white',
    cbar_kws={'label': 'Pearson r', 'shrink': 0.8},
)

ax.set_title(
    'Inter-Anchor Pearson Correlation\n'
    '8 GSVA Hallmark Anchors — TCGA-BRCA (n=1,031)',
    fontsize=13, fontweight='bold', pad=12
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=13)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=13)

plt.tight_layout()
plt.savefig('fig_anchor_correlation.png', dpi=200, bbox_inches='tight')
print("Saved: fig_anchor_correlation.png")