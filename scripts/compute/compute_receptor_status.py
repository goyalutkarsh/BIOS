"""
compute_receptor_status.py - Computes ER/PR/HER2 receptor status bio-anchors for TCGA-BRCA.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ── Gene definitions ──────────────────────────────────────────────────────────
RECEPTOR_SUPPORT_GENES = {
    "ER":   ["ESR1", "FOXA1", "GATA3", "TFF1", "TFF3", "AGR2"],
    "PR":   ["PGR", "FOXA1"],
    "HER2": ["ERBB2", "GRB7", "ERBB3"],
}

RECEPTOR_GENES = {
    "ER":   ["ESR1"],
    "PR":   ["PGR"],
    "HER2": ["ERBB2"],
}

CLINICAL_ER_COLS   = ["er_status_by_ihc", "ER_status", "er_status", "breast_carcinoma_estrogen_receptor_status"]
CLINICAL_PR_COLS   = ["pr_status_by_ihc", "PR_status", "pr_status", "breast_carcinoma_progesterone_receptor_status"]
CLINICAL_HER2_COLS = ["her2_status_by_ihc", "HER2_status", "her2_status", "lab_proc_her2_neu_immunohistochemistry_receptor_status"]

POSITIVE_LABELS = {"positive", "pos", "+", "1", "1.0"}
NEGATIVE_LABELS = {"negative", "neg", "-", "0", "0.0"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def find_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None

def parse_clinical_binary(series):
    normalized = series.astype(str).str.strip().str.lower()
    result = pd.Series(np.nan, index=series.index)
    result[normalized.isin(POSITIVE_LABELS)] = 1.0
    result[normalized.isin(NEGATIVE_LABELS)] = 0.0
    return result

def zscore_normalize(series):
    std = series.std()
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std

def strip_gene_id(index):
    """Strip |ENTREZ_ID suffix from gene names like FOXA1|2308 -> FOXA1"""
    return index.astype(str).str.strip().str.split("|").str[0]

def normalize_patient_id(index):
    """Normalize TCGA patient IDs to 15-char format TCGA-XX-XXXX-XX -> TCGA-XX-XXXX"""
    return index.astype(str).str.strip().str[:12]

# ── Main ──────────────────────────────────────────────────────────────────────

def compute_receptor_status(
    mrna_path, clinical_path=None, existing_anchors_path=None,
    output_path="bio_anchors_receptor_status.csv",
    use_multi_gene=True, verbose=True,
):
    print(f"\n{'='*60}")
    print("RECEPTOR STATUS BIO-ANCHOR COMPUTATION")
    print(f"{'='*60}\n")

    # ── 1. Load mRNA ──────────────────────────────────────────────────────────
    print(f"[1/5] Loading mRNA: {mrna_path}")
    mrna = pd.read_csv(mrna_path, index_col=0, sep="\t" if mrna_path.endswith(".tsv") else ",")
    print(f"  Raw shape: {mrna.shape}")

    # Strip |ENTREZ_ID from both index and columns before checking orientation
    mrna.index   = strip_gene_id(mrna.index)
    mrna.columns = mrna.columns.astype(str).str.strip()

    # Detect orientation: genes should be columns, patients should be rows
    # Check if key genes are in index (genes×patients) or columns (patients×genes)
    key_genes = ["FOXA1", "PGR", "ERBB3", "TFF1", "ERBB4"]
    genes_in_index   = sum(g in mrna.index   for g in key_genes)
    genes_in_columns = sum(g in mrna.columns for g in key_genes)
    print(f"  Key genes in index: {genes_in_index}, in columns: {genes_in_columns}")

    if genes_in_index > genes_in_columns:
        print(f"  Detected genes×patients orientation, transposing...")
        mrna = mrna.T
        # After transpose: index=patients, columns=genes (already stripped above)

    print(f"  Shape after orientation fix: {mrna.shape}  (patients × genes)")

    # Normalize patient IDs to 12-char TCGA format
    mrna.index = normalize_patient_id(mrna.index)
    mrna = mrna[~mrna.index.duplicated(keep="first")]  # keep one sample per patient

    # Check gene availability
    gene_set = RECEPTOR_SUPPORT_GENES if use_multi_gene else RECEPTOR_GENES
    for receptor, genes in gene_set.items():
        found = [g for g in genes if g in mrna.columns]
        print(f"  {receptor}: {len(found)}/{len(genes)} genes found → {found}")

    # ── 2. Load clinical ──────────────────────────────────────────────────────
    clinical_binary = {}
    if clinical_path:
        print(f"\n[2/5] Loading clinical data: {clinical_path}")
        clinical = pd.read_csv(clinical_path, index_col=0, sep="\t", on_bad_lines="skip", low_memory=False)
        clinical.index = normalize_patient_id(clinical.index)

        for receptor, candidates in zip(
            ["ER", "PR", "HER2"],
            [CLINICAL_ER_COLS, CLINICAL_PR_COLS, CLINICAL_HER2_COLS],
        ):
            col = find_column(clinical, candidates)
            if col:
                binary = parse_clinical_binary(clinical[col])
                n_pos = (binary == 1).sum()
                n_neg = (binary == 0).sum()
                n_na  = binary.isna().sum()
                print(f"  {receptor} clinical col '{col}': {n_pos} pos, {n_neg} neg, {n_na} missing")
                clinical_binary[receptor] = binary
            else:
                print(f"  [WARN] No clinical column found for {receptor}")
    else:
        print("\n[2/5] No clinical path provided — skipping validation")

    # ── 3. Compute scores ─────────────────────────────────────────────────────
    print("\n[3/5] Computing continuous expression scores...")
    scores = {}
    genes_used = {}

    for receptor in ["ER", "PR", "HER2"]:
        gene_list = (RECEPTOR_SUPPORT_GENES if use_multi_gene else RECEPTOR_GENES)[receptor]
        available = [g for g in gene_list if g in mrna.columns]

        if not available:
            print(f"  [ERROR] No genes available for {receptor}! Filling with zeros.")
            scores[receptor] = pd.Series(0.0, index=mrna.index)
            genes_used[receptor] = []
            continue

        score = mrna[available].mean(axis=1)
        scores[receptor] = score
        genes_used[receptor] = available
        print(f"  {receptor}: mean of {available}  →  mean={score.mean():.3f}, std={score.std():.3f}")

    # ── 4. Validate ───────────────────────────────────────────────────────────
    print("\n[4/5] Validating against clinical labels (ROC AUC)...")
    aucs = {}
    for receptor, score in scores.items():
        if receptor not in clinical_binary:
            print(f"  {receptor}: No clinical labels available")
            continue
        try:
            from sklearn.metrics import roc_auc_score
            common = score.index.intersection(clinical_binary[receptor].dropna().index)
            if len(common) < 50:
                print(f"  [WARN] Only {len(common)} overlapping patients for {receptor}, skipping AUC")
                continue
            auc = roc_auc_score(clinical_binary[receptor].loc[common], score.loc[common])
            if auc < 0.5:
                auc = 1.0 - auc
            aucs[receptor] = auc
            status = "✅" if auc >= 0.80 else ("⚠️" if auc >= 0.70 else "❌")
            print(f"  {receptor} AUC: {auc:.3f} {status}  (n={len(common)})")
        except Exception as e:
            print(f"  [WARN] AUC failed for {receptor}: {e}")

    # ── 5. Normalize and save ─────────────────────────────────────────────────
    print("\n[5/5] Normalizing and saving...")

    result = pd.DataFrame({
        "patient_id": mrna.index,
        "ER_score":   zscore_normalize(scores["ER"]).values,
        "PR_score":   zscore_normalize(scores["PR"]).values,
        "HER2_score": zscore_normalize(scores["HER2"]).values,
    })

    if existing_anchors_path:
        print(f"  Merging with existing anchors: {existing_anchors_path}")
        existing = pd.read_csv(existing_anchors_path)
        pid_col = existing.columns[0]
        existing = existing.rename(columns={pid_col: "patient_id"})
        existing["patient_id"] = normalize_patient_id(existing["patient_id"])
        result["patient_id"]   = normalize_patient_id(result["patient_id"])

        print(f"  Existing patients sample: {existing['patient_id'].iloc[:3].tolist()}")
        print(f"  mRNA patients sample:     {result['patient_id'].iloc[:3].tolist()}")

        merged = existing.merge(result, on="patient_id", how="inner")
        print(f"  Merge: {len(existing)} existing × {len(result)} mRNA → {len(merged)} matched")
        result = merged

    result.to_csv(output_path, index=False)
    print(f"\n  Saved to: {output_path}")
    print(f"  Shape: {result.shape}")
    print(f"  Columns: {list(result.columns)}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Patients: {len(result)}")
    print(f"Genes used:")
    for r, g in genes_used.items():
        print(f"  {r}: {g}")
    if aucs:
        print(f"Validation AUCs:")
        for r, auc in aucs.items():
            status = "✅" if auc >= 0.80 else ("⚠️" if auc >= 0.70 else "❌")
            print(f"  {r}: {auc:.3f} {status}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Compute ER/PR/HER2 receptor status bio-anchors")
    parser.add_argument("--mrna",      required=True)
    parser.add_argument("--clinical",  default=None)
    parser.add_argument("--existing",  default=None)
    parser.add_argument("--output",    default="bio_anchors_receptor_status.csv")
    parser.add_argument("--single-gene", action="store_true")
    parser.add_argument("--quiet",     action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compute_receptor_status(
        mrna_path=args.mrna,
        clinical_path=args.clinical,
        existing_anchors_path=args.existing,
        output_path=args.output,
        use_multi_gene=not args.single_gene,
        verbose=not args.quiet,
    )