import argparse
import pandas as pd
from scipy.stats import zscore

PATHWAY_ANCHORS = {
    "PI3K_score":      ["PIK3CA", "AKT1", "MTOR", "PTEN"],
    "p53_score":       ["TP53", "MDM2", "CDKN1A"],
    "MYC_score":       ["MYC", "MYCN"],
    "MAPK_score":      ["KRAS", "BRAF", "EGFR"],
    "WNT_score":       ["CTNNB1", "APC"],
    "hypoxia_score":   ["HIF1A", "VEGFA", "LDHA", "SLC2A1"],
    "cellcycle_score": ["CCND1", "CDK4", "CDK6", "RB1", "CCNB1", "CDC20"],
    "apoptosis_score": ["BCL2", "BAX", "CASP3", "CASP9"],
    "angio_score":     ["VEGFA", "PECAM1", "ANGPT2"],
    "dna_repair_score":["BRCA1", "BRCA2", "RAD51", "PARP1"],
}
SUPPRESSORS = {"PTEN", "APC", "RB1"}

def normalize_patient_id(ids):
    return [str(i)[:12] for i in ids]

def compute_pathway_score(mrna, genes):
    available = [g for g in genes if g in mrna.columns]
    scores = []
    for g in available:
        s = mrna[g].copy()
        if g in SUPPRESSORS:
            s = -s
        scores.append(s)
    return pd.concat(scores, axis=1).mean(axis=1), available

parser = argparse.ArgumentParser()
parser.add_argument("--mrna", type=str, required=True)
parser.add_argument("--existing", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()

mrna = pd.read_csv(args.mrna, sep="\t", index_col=0)
if mrna.shape[0] < mrna.shape[1]:
    mrna = mrna.T
mrna.index = normalize_patient_id(mrna.index)
mrna = mrna[~mrna.index.duplicated(keep="first")]
print("mRNA shape:", mrna.shape)

pathway_scores = {}
genes_used = {}
for name, genes in PATHWAY_ANCHORS.items():
    score, used = compute_pathway_score(mrna, genes)
    pathway_scores[name] = score
    genes_used[name] = used
    print(name, used, "mean=", round(score.mean(), 3))

pathway_df = pd.DataFrame(pathway_scores)
pathway_df.index.name = "patient_id"

existing = pd.read_csv(args.existing)
existing["patient_id"] = normalize_patient_id(existing["patient_id"])
existing = existing.set_index("patient_id")
merged = existing.join(pathway_df, how="inner")
print("Merged:", len(merged), "patients")

for col in PATHWAY_ANCHORS.keys():
    merged[col] = zscore(merged[col])

merged = merged.reset_index()
merged.to_csv(args.output, index=False)
print("Saved:", args.output)
print(merged.describe().round(3).to_string())
