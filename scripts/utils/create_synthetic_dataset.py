import numpy as np
import pandas as pd
import argparse

def generate_synthetic_data(n_samples=1000, n_clusters=4, n_features=500, noise_level=0.1, seed=42):
    """
    Generate synthetic multi-omics-like data with known cluster structure and bio-anchors
    
    Args:
        n_samples: Number of samples (patients)
        n_clusters: Number of true clusters (subtypes)
        n_features: Dimensionality of feature space
        noise_level: Amount of noise to add (0=clean, 1=very noisy)
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    
    # Create ground truth cluster assignments (balanced)
    samples_per_cluster = n_samples // n_clusters
    true_labels = np.repeat(np.arange(n_clusters), samples_per_cluster)
    
    # Handle remainder
    if n_samples % n_clusters != 0:
        remainder = n_samples % n_clusters
        true_labels = np.concatenate([true_labels, np.arange(remainder)])
    
    # Shuffle to make it more realistic
    shuffle_idx = np.random.permutation(n_samples)
    true_labels = true_labels[shuffle_idx]
    
    print(f"Generated {n_samples} samples with {n_clusters} clusters")
    print(f"Cluster distribution: {np.bincount(true_labels)}")
    
    # Define cluster-specific bio-anchor patterns
    cluster_bioanchor_patterns = {
        0: {  # "Basal-like" phenotype
            'immune_score': (0.7, 1.0),
            'HRD_score': (60, 100),
            'TMB': (15, 20),
            'proliferation': (0.7, 0.9),
            'PI3K_pathway': (1.0, 2.5),
            'MAPK_pathway': (-0.5, 0.5),
            'p53_pathway': (1.5, 2.5),
            'WNT_pathway': (-1.0, 0.0),
            'MYC_pathway': (1.0, 2.0),
            'EMT_score': (0.6, 0.9),
            'angiogenesis': (0.5, 0.8),
            'hypoxia': (0.6, 0.9),
            'stromal_score': (0.3, 0.6),
            'tumor_purity': (0.6, 0.8),
            'TP53_mutation': (0.7, 1.0)  # High mutation rate
        },
        1: {  # "Luminal A" phenotype
            'immune_score': (0.2, 0.4),
            'HRD_score': (10, 30),
            'TMB': (2, 6),
            'proliferation': (0.1, 0.3),
            'PI3K_pathway': (-1.0, 0.5),
            'MAPK_pathway': (-1.5, -0.5),
            'p53_pathway': (-1.0, 0.0),
            'WNT_pathway': (0.5, 1.5),
            'MYC_pathway': (-1.0, 0.0),
            'EMT_score': (0.1, 0.3),
            'angiogenesis': (0.2, 0.4),
            'hypoxia': (0.1, 0.3),
            'stromal_score': (0.5, 0.8),
            'tumor_purity': (0.7, 0.9),
            'TP53_mutation': (0.1, 0.3)
        },
        2: {  # "Her2-enriched" phenotype
            'immune_score': (0.4, 0.7),
            'HRD_score': (30, 60),
            'TMB': (8, 15),
            'proliferation': (0.5, 0.8),
            'PI3K_pathway': (0.5, 1.5),
            'MAPK_pathway': (1.0, 2.0),
            'p53_pathway': (0.5, 1.5),
            'WNT_pathway': (-0.5, 0.5),
            'MYC_pathway': (0.5, 1.5),
            'EMT_score': (0.3, 0.6),
            'angiogenesis': (0.5, 0.7),
            'hypoxia': (0.4, 0.6),
            'stromal_score': (0.4, 0.7),
            'tumor_purity': (0.6, 0.8),
            'TP53_mutation': (0.5, 0.8)
        },
        3: {  # "Luminal B" phenotype
            'immune_score': (0.3, 0.6),
            'HRD_score': (20, 50),
            'TMB': (5, 12),
            'proliferation': (0.4, 0.7),
            'PI3K_pathway': (0.0, 1.0),
            'MAPK_pathway': (-0.5, 0.5),
            'p53_pathway': (0.0, 1.0),
            'WNT_pathway': (0.0, 1.0),
            'MYC_pathway': (0.0, 1.0),
            'EMT_score': (0.2, 0.5),
            'angiogenesis': (0.3, 0.6),
            'hypoxia': (0.3, 0.6),
            'stromal_score': (0.4, 0.7),
            'tumor_purity': (0.6, 0.8),
            'TP53_mutation': (0.3, 0.6)
        }
    }
    
    # Generate bio-anchors based on cluster membership
    bio_anchor_names = list(cluster_bioanchor_patterns[0].keys())
    bio_anchors = np.zeros((n_samples, len(bio_anchor_names)))
    
    for i, label in enumerate(true_labels):
        pattern = cluster_bioanchor_patterns[label % n_clusters]  # Handle extra clusters
        for j, anchor_name in enumerate(bio_anchor_names):
            low, high = pattern[anchor_name]
            bio_anchors[i, j] = np.random.uniform(low, high)
    
    # Add noise to bio-anchors
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, bio_anchors.shape)
        bio_anchors = bio_anchors + noise
        # Clip to reasonable ranges
        bio_anchors = np.clip(bio_anchors, -3, 3)  # For pathway scores
    
    # Generate features that correlate with bio-anchors
    # Some features are strongly correlated, some weakly, some random
    features = np.zeros((n_samples, n_features))
    
    # First 100 features: strongly correlated with bio-anchors
    for i in range(min(100, n_features)):
        anchor_idx = i % len(bio_anchor_names)
        features[:, i] = bio_anchors[:, anchor_idx] + np.random.normal(0, 0.1, n_samples)
    
    # Next 200 features: weakly correlated with bio-anchors + noise
    for i in range(100, min(300, n_features)):
        anchor_idx = i % len(bio_anchor_names)
        features[:, i] = 0.3 * bio_anchors[:, anchor_idx] + np.random.normal(0, 0.5, n_samples)
    
    # Remaining features: cluster-specific patterns but not directly from bio-anchors
    for i in range(300, n_features):
        cluster_means = np.random.randn(n_clusters) * 2
        for j in range(n_samples):
            cluster = true_labels[j]
            features[j, i] = cluster_means[cluster] + np.random.normal(0, 0.5)
    
    # Add global noise
    if noise_level > 0:
        features += np.random.normal(0, noise_level * 0.5, features.shape)
    
    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    return features, bio_anchors, true_labels, bio_anchor_names


def save_synthetic_data(features, bio_anchors, true_labels, bio_anchor_names, cancer_type="SYNTHETIC"):
    """
    Save synthetic data in the same format as real data
    """
    n_samples = features.shape[0]
    patient_ids = [f"PATIENT_{i:04d}" for i in range(n_samples)]
    
    # Save features as .fea files (mimicking real data structure)
    # Split features into 4 "omics" types
    n_features = features.shape[1]
    split_points = [0, n_features//4, n_features//2, 3*n_features//4, n_features]
    omics_names = ['CN', 'meth', 'miRNA', 'rna']
    
    import os
    fea_dir = f'../subtype_file/{cancer_type}'
    os.makedirs(fea_dir, exist_ok=True)
    
    for i, omics in enumerate(omics_names):
        start_idx = split_points[i]
        end_idx = split_points[i+1]
        omics_features = features[:, start_idx:end_idx].T
        
        # Create DataFrame with gene names as rows, patients as columns
        gene_names = [f"{omics}_feature_{j:04d}" for j in range(omics_features.shape[0])]
        df = pd.DataFrame(omics_features, index=gene_names, columns=patient_ids)
        
        filepath = f'{fea_dir}/{omics}.fea'
        df.to_csv(filepath, sep=',')
        print(f"Saved {filepath}: {df.shape}")
    
    # Save bio-anchors
    bio_df = pd.DataFrame(bio_anchors, columns=bio_anchor_names)
    bio_df.insert(0, 'patient_id', patient_ids)
    bio_filepath = f'bio_anchors_{cancer_type}.csv'
    bio_df.to_csv(bio_filepath, index=False)
    print(f"Saved {bio_filepath}: {bio_df.shape}")
    
    # Save ground truth labels
    labels_df = pd.DataFrame({
        'patient_id': patient_ids,
        'true_cluster': true_labels
    })
    labels_filepath = f'ground_truth_{cancer_type}.csv'
    labels_df.to_csv(labels_filepath, index=False)
    print(f"Saved {labels_filepath}: {labels_df.shape}")
    
    print(f"\n✅ Synthetic data generation complete!")
    print(f"   - Features: {features.shape} saved to {fea_dir}/")
    print(f"   - Bio-anchors: {bio_anchors.shape} saved to {bio_filepath}")
    print(f"   - Ground truth: {true_labels.shape} saved to {labels_filepath}")
    print(f"\nCluster distribution:")
    for cluster_id in range(len(np.bincount(true_labels))):
        count = np.sum(true_labels == cluster_id)
        print(f"   Cluster {cluster_id}: {count} samples ({100*count/n_samples:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic cancer subtyping data")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--n_clusters", type=int, default=4, help="Number of clusters")
    parser.add_argument("--n_features", type=int, default=1000, help="Feature dimensionality")
    parser.add_argument("--noise_level", type=float, default=0.1, help="Noise level (0-1)")
    parser.add_argument("--cancer_type", type=str, default="SYNTHETIC", help="Cancer type name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    print("="*60)
    print("SYNTHETIC DATA GENERATOR")
    print("="*60)
    print(f"Parameters:")
    print(f"  - Samples: {args.n_samples}")
    print(f"  - Clusters: {args.n_clusters}")
    print(f"  - Features: {args.n_features}")
    print(f"  - Noise level: {args.noise_level}")
    print(f"  - Seed: {args.seed}")
    print("="*60)
    
    # Generate data
    features, bio_anchors, true_labels, bio_anchor_names = generate_synthetic_data(
        n_samples=args.n_samples,
        n_clusters=args.n_clusters,
        n_features=args.n_features,
        noise_level=args.noise_level,
        seed=args.seed
    )
    
    # Save data
    save_synthetic_data(features, bio_anchors, true_labels, bio_anchor_names, args.cancer_type)
