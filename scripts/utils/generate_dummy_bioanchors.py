import pandas as pd
import numpy as np

# Generate dummy bio-anchors for BRCA dataset
# In reality, these would be computed from the .fea files

# Load one of the feature files to get patient IDs
fea_file = '../subtype_file/BRCA/CN.fea'
fea_data = pd.read_csv(fea_file, header=0, index_col=0, sep=',')
patient_ids = fea_data.columns.tolist()

n_patients = len(patient_ids)
print(f"Generating bio-anchors for {n_patients} patients")

# Generate 15 bio-anchors (all synthetic for now)
np.random.seed(42)
bio_anchors = {
    'patient_id': patient_ids,
    'immune_score': np.random.uniform(0, 1, n_patients),
    'HRD_score': np.random.uniform(0, 100, n_patients),
    'TMB': np.random.uniform(0, 20, n_patients),
    'proliferation': np.random.uniform(0, 1, n_patients),
    'PI3K_pathway': np.random.uniform(-2, 2, n_patients),
    'MAPK_pathway': np.random.uniform(-2, 2, n_patients),
    'p53_pathway': np.random.uniform(-2, 2, n_patients),
    'WNT_pathway': np.random.uniform(-2, 2, n_patients),
    'MYC_pathway': np.random.uniform(-2, 2, n_patients),
    'EMT_score': np.random.uniform(0, 1, n_patients),
    'angiogenesis': np.random.uniform(0, 1, n_patients),
    'hypoxia': np.random.uniform(0, 1, n_patients),
    'stromal_score': np.random.uniform(0, 1, n_patients),
    'tumor_purity': np.random.uniform(0.3, 1.0, n_patients),
    'TP53_mutation': np.random.randint(0, 2, n_patients)
}

df = pd.DataFrame(bio_anchors)
df.to_csv('bio_anchors_BRCA.csv', index=False)
print(f"Saved bio-anchors to bio_anchors_BRCA.csv")
print(f"Shape: {df.shape}")
print(df.head())
