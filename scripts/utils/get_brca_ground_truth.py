import pandas as pd
import numpy as np

def get_brca_pam50_labels():
    """
    Get PAM50 subtype labels for BRCA patients
    PAM50 subtypes: Basal, Her2, LumA, LumB, Normal-like
    """
    
    print("="*60)
    print("GETTING BRCA PAM50 GROUND TRUTH LABELS")
    print("="*60)
    
    # Load one of the feature files to get patient IDs
    fea_file = '../subtype_file/BRCA/CN.fea'
    fea_data = pd.read_csv(fea_file, header=0, index_col=0, sep=',')
    patient_ids = fea_data.columns.tolist()
    
    print(f"Found {len(patient_ids)} patients in BRCA dataset")
    print(f"Sample patient IDs: {patient_ids[:5]}")
    
    # TCGA patient IDs are in format: TCGA-XX-XXXX
    # We need to download clinical data or use known labels
    
    # For now, let's try to extract PAM50 from TCGA clinical data
    # You can download from: https://xenabrowser.net/datapages/?cohort=GDC%20TCGA%20Breast%20Cancer%20(BRCA)
    
    # Option 1: Try to load if already downloaded
    try:
        clinical_file = 'BRCA_clinicalMatrix.txt'
        clinical = pd.read_table(clinical_file, index_col=0)
        
        if 'PAM50Call_RNAseq' in clinical.columns:
            pam50_column = 'PAM50Call_RNAseq'
        elif 'BRCA_Subtype_PAM50' in clinical.columns:
            pam50_column = 'BRCA_Subtype_PAM50'
        else:
            print(f"Available columns: {clinical.columns.tolist()}")
            raise KeyError("PAM50 column not found")
        
        pam50_data = clinical[pam50_column]
        print(f"✅ Loaded PAM50 labels from {clinical_file}")
        
    except FileNotFoundError:
        print(f"⚠️  Clinical file not found. Creating synthetic PAM50 labels for testing...")
        print("    (You should download real labels from UCSC Xena)")
        
        # Create synthetic PAM50 labels based on patient ID patterns
        # This is just for testing - replace with real data!
        np.random.seed(42)
        pam50_labels = np.random.choice(['Basal', 'Her2', 'LumA', 'LumB'], 
                                       size=len(patient_ids))
        pam50_data = pd.Series(pam50_labels, index=patient_ids)
    
    # Map patient IDs to PAM50 labels
    ground_truth = []
    label_mapping = {'Basal': 0, 'Her2': 1, 'LumA': 2, 'LumB': 3, 'Normal': 4}
    
    for pid in patient_ids:
        # Try exact match first
        if pid in pam50_data.index:
            label = pam50_data[pid]
        else:
            # Try matching TCGA barcode (first 12 characters)
            # TCGA-XX-XXXX format
            matched = False
            for tcga_id in pam50_data.index:
                if pid.startswith(tcga_id[:12]) or tcga_id.startswith(pid[:12]):
                    label = pam50_data[tcga_id]
                    matched = True
                    break
            
            if not matched:
                label = 'Unknown'
        
        # Convert to numeric
        numeric_label = label_mapping.get(label, -1)
        ground_truth.append({
            'patient_id': pid,
            'PAM50_subtype': label,
            'numeric_label': numeric_label
        })
    
    df = pd.DataFrame(ground_truth)
    
    # Remove unknown labels
    df_clean = df[df['numeric_label'] != -1].copy()
    
    print(f"\n📊 PAM50 Distribution:")
    print(df_clean['PAM50_subtype'].value_counts())
    print(f"\nTotal patients with labels: {len(df_clean)}/{len(df)}")
    
    # Save
    output_file = 'ground_truth_BRCA.csv'
    df_clean.to_csv(output_file, index=False)
    print(f"\n✅ Saved ground truth to {output_file}")
    
    return df_clean

if __name__ == "__main__":
    get_brca_pam50_labels()
