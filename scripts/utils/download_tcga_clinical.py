import pandas as pd
import urllib.request
import os

def download_brca_clinical():
    """
    Download BRCA clinical data from UCSC Xena with PAM50 labels
    """
    print("="*60)
    print("DOWNLOADING TCGA BRCA CLINICAL DATA")
    print("="*60)
    
    # UCSC Xena URL for BRCA clinical data with PAM50
    url = "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.BRCA.sampleMap%2FBRCA_clinicalMatrix"
    
    output_file = "BRCA_clinicalMatrix.txt"
    
    print(f"\n📥 Downloading from UCSC Xena...")
    print(f"   URL: {url}")
    
    try:
        urllib.request.urlretrieve(url, output_file)
        print(f"✅ Downloaded to {output_file}")
        
        # Check file
        df = pd.read_table(output_file, nrows=5)
        print(f"\n📊 Preview:")
        print(f"   Rows: {len(df)}")
        print(f"   Columns: {len(df.columns)}")
        print(f"   Sample columns: {df.columns[:10].tolist()}")
        
        # Look for PAM50 column
        pam50_cols = [col for col in df.columns if 'PAM50' in col or 'subtype' in col.lower()]
        if pam50_cols:
            print(f"\n✅ Found PAM50 columns: {pam50_cols}")
        else:
            print(f"\n⚠️  No obvious PAM50 column found")
            print(f"   All columns: {df.columns.tolist()}")
        
        return output_file
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print(f"\n📝 Manual download instructions:")
        print(f"   1. Go to: https://xenabrowser.net/datapages/?cohort=GDC%20TCGA%20Breast%20Cancer%20(BRCA)")
        print(f"   2. Find 'Phenotype' → 'BRCA_clinicalMatrix'")
        print(f"   3. Download and save as: {output_file}")
        return None

if __name__ == "__main__":
    download_brca_clinical()
