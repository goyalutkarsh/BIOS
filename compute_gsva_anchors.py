"""
compute_gsva_anchors.py
=======================
Computes GSVA bio-anchors from TCGA mRNA expression and saves a CSV
that plugs directly into BIOS dataloader via --bio_anchor_file.

Output CSV format:
    patient_id, HALLMARK_INFLAMMATORY_RESPONSE, HALLMARK_EMT, ...
    TCGA-A2-A04X,  0.23, -0.11, ...
    TCGA-E2-A14N,  1.45,  0.67, ...

Row order doesn't matter — dataloader aligns by patient_id.

USAGE (on HPC):
    PYBIN=/gpfs/home/ug25b/.conda/envs/bios/bin/python

    # Step 1 — find your mRNA file path, then run:
    $PYBIN compute_gsva_anchors.py \
        --rna_file ../subtype_file/fea/BRCA/rna.fea \
        --cancer_type BRCA \
        --mode M8 \
        --out_file bio_anchors_BRCA_gsva_M8.csv

    # Step 2 — verify the output:
    $PYBIN compute_gsva_anchors.py --verify bio_anchors_BRCA_gsva_M8.csv

    # Step 3 — train BIOS with new anchors (update --bio_dim and --bio_anchor_file):
    $PYBIN train_bioanchor.py \
        --head_type linear \
        --bio_dim 8 \
        --cancer_type BRCA \
        --cluster_number 5 \
        --bio_anchor_file bio_anchors_BRCA_gsva_M8.csv
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd

# gsva_anchors.py must be in the same directory
from gsva_anchors import load_or_compute_gsva_anchors, validate_gsva_anchors, CONFIGS


def load_rna_fea(rna_file):
    """
    Load rna.fea file — same format your dataloader uses.

    rna.fea format (from your dataloader):
        rows    = genes
        columns = patients (TCGA sample IDs)
        sep     = ','
        header  = 0, index_col = 0

    Returns
    -------
    expr_df : pd.DataFrame, shape (n_genes, n_samples)
              index = gene symbols, columns = patient IDs
    """
    print(f"[INFO] Loading mRNA from: {rna_file}")
    sep = '\t' if rna_file.endswith('.tsv') or rna_file.endswith('.tsv.gz') else ','
    expr_df = pd.read_csv(rna_file, header=0, index_col=0, sep=sep)
    expr_df.index = expr_df.index.str.split('|').str[0] # Strip Entrez IDs from gene names (e.g. "TP53|7157" → "TP53")

    print(f"[INFO] Loaded shape: {expr_df.shape}")
    print(f"[INFO] Sample gene names (index): {expr_df.index[:5].tolist()}")
    print(f"[INFO] Sample patient IDs (columns): {expr_df.columns[:3].tolist()}")

    # Confirm orientation: rows=genes, columns=patients
    # If more columns than rows, it's transposed — fix it
    if expr_df.shape[1] > expr_df.shape[0]:
        print(f"[INFO] Transposing to genes x patients: {expr_df.T.shape}")
        expr_df = expr_df.T

    return expr_df


def save_anchor_csv(anchors_df, out_file):
    """
    Save GSVA anchors as CSV with patient_id as first column.
    This matches exactly what your dataloader expects.

    anchors_df : pd.DataFrame (n_samples x n_anchors)
                 index = patient IDs (TCGA sample IDs)
    """
    out_df = anchors_df.copy()
    # Keep only primary tumor samples (-01 suffix) before truncating
    # This prevents duplicates when multiple sample types exist per patient
    out_df = out_df[out_df.index.str.endswith('-01')]
    out_df.index = out_df.index.str[:12]  # TCGA-AR-A5QQ-01 → TCGA-AR-A5QQ
    out_df.index.name = 'patient_id'
    out_df = out_df.reset_index()
    out_df.to_csv(out_file, index=False)
    print(f"[INFO] Saved: {out_file}  shape={out_df.shape}")
    print(f"[INFO] Columns: {out_df.columns.tolist()}")
    print(f"\nFirst 3 rows:")
    print(out_df.head(3).to_string(index=False))


def verify_csv(csv_file, cn_file=None):
    """
    Verify the output CSV is in the right format and
    optionally check patient_id alignment with CN.fea.
    """
    print(f"\n[VERIFY] Reading {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"  Shape      : {df.shape}")
    print(f"  Columns    : {df.columns.tolist()}")
    print(f"  patient_id sample: {df['patient_id'].head(3).tolist()}")
    print(f"  Value range: [{df.iloc[:,1:].values.min():.3f}, {df.iloc[:,1:].values.max():.3f}]")
    print(f"  Any NaN    : {df.isnull().any().any()}")

    if cn_file and os.path.exists(cn_file):
        cn = pd.read_csv(cn_file, header=0, index_col=0, sep=',')
        patient_ids = cn.columns.tolist()
        anchor_ids  = df['patient_id'].tolist()
        overlap     = len(set(patient_ids) & set(anchor_ids))
        print(f"\n  CN.fea patients   : {len(patient_ids)}")
        print(f"  Anchor patients   : {len(anchor_ids)}")
        print(f"  Overlap           : {overlap}")
        if overlap == len(patient_ids):
            print("  [OK] All patients matched.")
        else:
            missing = set(patient_ids) - set(anchor_ids)
            print(f"  [WARN] {len(missing)} patients in CN.fea not in anchors: {list(missing)[:5]}")
    else:
        print(f"  [SKIP] CN.fea check (not provided or not found)")

    print("\n[VERIFY] Done.")


def main():
    parser = argparse.ArgumentParser(description="Compute GSVA bio-anchors for BIOS")

    parser.add_argument("--rna_file",    type=str,
                        help="Path to rna.fea file (genes x patients, comma-separated)")
    parser.add_argument("--cancer_type", type=str, default="BRCA",
                        help="Cancer type label (used for cache filename only)")
    parser.add_argument("--mode",        type=str, default="M8",
                        choices=list(CONFIGS.keys()),
                        help="Anchor config: M4 | M5_pi3k | M5_dna | M6 | M7 | M8")
    parser.add_argument("--out_file",    type=str,
                        help="Output CSV path (e.g. bio_anchors_BRCA_gsva_M8.csv)")
    parser.add_argument("--processes",   type=int, default=4,
                        help="Parallel processes for ssGSEA (use 1 for debugging)")
    parser.add_argument("--force",       action="store_true",
                        help="Recompute even if cache exists")
    parser.add_argument("--validate",    action="store_true",
                        help="Print correlation matrix after computing")
    parser.add_argument("--verify",      type=str, default=None,
                        help="Verify an existing CSV (skips computation). "
                             "Pass path to CSV: --verify bio_anchors_BRCA_gsva_M8.csv")
    parser.add_argument("--cn_file",     type=str, default=None,
                        help="Path to CN.fea to verify patient_id alignment")

    args = parser.parse_args()

    # ── verify-only mode ─────────────────────────────────────────
    if args.verify:
        verify_csv(args.verify, args.cn_file)
        return

    # ── compute mode ─────────────────────────────────────────────
    if not args.rna_file:
        print("ERROR: --rna_file required. Run with --help for usage.")
        sys.exit(1)
    if not args.out_file:
        args.out_file = f"bio_anchors_{args.cancer_type}_gsva_{args.mode}.csv"
        print(f"[INFO] --out_file not set, using: {args.out_file}")

    # Load mRNA expression
    expr_df = load_rna_fea(args.rna_file)

    # Cache path — keeps one pkl per cancer/mode combo
    cache_path = f"gsva_cache_{args.cancer_type}_{args.mode}.pkl"

    # Compute GSVA anchors (cached after first run)
    anchors = load_or_compute_gsva_anchors(
        expr_df,
        cache_path     = cache_path,
        mode           = args.mode,
        force_recompute= args.force,
        processes      = args.processes,
        verbose        = True,
    )

    print(f"\n[INFO] Anchor matrix: {anchors.shape}")

    # Optional validation
    if args.validate:
        validate_gsva_anchors(anchors)

    # Save as CSV in BIOS dataloader format
    save_anchor_csv(anchors, args.out_file)

    # Auto-verify after saving
    verify_csv(args.out_file, args.cn_file)

    print(f"""
Done. Next steps:
  1. Update your SLURM script:
       --bio_anchor_file {args.out_file} \\
       --bio_dim {anchors.shape[1]}

  2. That's it. network.py and train_bioanchor.py need no changes.
""")


if __name__ == "__main__":
    main()
