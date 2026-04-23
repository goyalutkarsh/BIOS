import pandas as pd

# Load RNA data
rna_file = '../subtype_file/BRCA/rna.fea'
rna_data = pd.read_csv(rna_file, header=0, index_col=0, sep=',')

print("Sample gene names (first 20):")
print(rna_data.index[:20].tolist())

print("\n\nSearching for common cancer genes...")
search_genes = ['TP53', 'BRCA1', 'BRCA2', 'MKI67', 'ESR1', 'ERBB2', 'CD8A']

for gene in search_genes:
    matches = [g for g in rna_data.index if gene in g.upper()]
    if matches:
        print(f"{gene}: Found {len(matches)} matches - {matches[:3]}")
    else:
        print(f"{gene}: Not found")
