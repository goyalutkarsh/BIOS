import torch
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
from network import Network
from ae import AE
from dataloader import get_feature
import argparse

def load_model(model_path, feature_dim, cluster_num, device):
    """Load trained model"""
    ae = AE(hid_dim=feature_dim, bio_dim=15)
    model = network.Network(ae, feature_dim, cluster_num)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['net'])
    model = model.to(device)
    model.eval()
    
    return model

def get_predictions(model, dataloader, device):
    """Get cluster predictions and embeddings from model"""
    all_predictions = []
    all_embeddings = []
    
    with torch.no_grad():
        for batch_data in dataloader:
            if len(batch_data) == 2:
                x, _ = batch_data
            else:
                x = batch_data[0]
            
            x = x.float().to(device)
            
            # Get embeddings and cluster assignments
            z, z_bio, z_novel = model.ae(x)
            c, h = model.forward_cluster(x)
            
            all_predictions.extend(c.cpu().numpy())
            all_embeddings.extend(z.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_embeddings)

def evaluate_clustering(pred_labels, true_labels):
    """Compute clustering metrics"""
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    v_measure = v_measure_score(true_labels, pred_labels)
    
    return {
        'ARI': ari,
        'NMI': nmi,
        'V-measure': v_measure
    }

def plot_confusion_matrix(pred_labels, true_labels, cancer_type):
    """Plot confusion matrix"""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(true_labels, pred_labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'Pred {i}' for i in range(cm.shape[1])],
                yticklabels=[f'True {i}' for i in range(cm.shape[0])])
    plt.xlabel('Predicted Cluster')
    plt.ylabel('True Cluster')
    plt.title(f'Confusion Matrix - {cancer_type}')
    plt.tight_layout()
    plt.savefig(f'results/{cancer_type}_confusion_matrix.png', dpi=150)
    print(f"Saved confusion matrix to results/{cancer_type}_confusion_matrix.png")

def plot_cluster_distribution(pred_labels, true_labels, cancer_type):
    """Plot cluster distribution comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # True labels distribution
    true_counts = np.bincount(true_labels)
    ax1.bar(range(len(true_counts)), true_counts, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('Count')
    ax1.set_title('True Cluster Distribution')
    ax1.set_xticks(range(len(true_counts)))
    
    # Predicted labels distribution
    pred_counts = np.bincount(pred_labels)
    ax2.bar(range(len(pred_counts)), pred_counts, color='coral', alpha=0.7)
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Count')
    ax2.set_title('Predicted Cluster Distribution')
    ax2.set_xticks(range(len(pred_counts)))
    
    plt.tight_layout()
    plt.savefig(f'results/{cancer_type}_cluster_distribution.png', dpi=150)
    print(f"Saved distribution plot to results/{cancer_type}_cluster_distribution.png")

def plot_embeddings_tsne(embeddings, true_labels, pred_labels, cancer_type):
    """Plot t-SNE visualization of embeddings"""
    from sklearn.manifold import TSNE
    
    print("Computing t-SNE (this may take a minute)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Color by true labels
    scatter1 = ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                          c=true_labels, cmap='tab10', alpha=0.6, s=20)
    ax1.set_title('t-SNE colored by True Labels')
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    plt.colorbar(scatter1, ax=ax1, label='True Cluster')
    
    # Color by predicted labels
    scatter2 = ax2.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                          c=pred_labels, cmap='tab10', alpha=0.6, s=20)
    ax2.set_title('t-SNE colored by Predicted Labels')
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    plt.colorbar(scatter2, ax=ax2, label='Predicted Cluster')
    
    plt.tight_layout()
    plt.savefig(f'results/{cancer_type}_tsne.png', dpi=150)
    print(f"Saved t-SNE plot to results/{cancer_type}_tsne.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer_type", type=str, default="SYNTHETIC")
    parser.add_argument("--model_path", type=str, default="save/model/checkpoint_10.tar")
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    
    args = parser.parse_args()
    
    print("="*60)
    print("CLUSTERING EVALUATION")
    print("="*60)
    
    # Load ground truth
    ground_truth_file = f'data/ground_truth/ground_truth_{args.cancer_type}.csv'
    try:
        gt_df = pd.read_csv(ground_truth_file)
        
        # Handle different column names
        if 'true_cluster' in gt_df.columns:
            label_col = 'true_cluster'
        elif 'numeric_label' in gt_df.columns:
            label_col = 'numeric_label'
        else:
            print(f"Available columns: {gt_df.columns.tolist()}")
            raise KeyError("No ground truth label column found")
        
        print(f"✅ Loaded ground truth from {ground_truth_file}")
        print(f"   Total patients with labels: {len(gt_df)}")
        
    except FileNotFoundError:
        print(f"❌ Ground truth file not found: {ground_truth_file}")
        print("   Evaluation requires ground truth labels.")
        exit(1)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cluster_num = len(gt_df[label_col].unique())
    
    # Load model
    print(f"\n📂 Loading model from {args.model_path}...")
    model = load_model(args.model_path, args.feature_dim, cluster_num, device)
    print("✅ Model loaded successfully")
    
    # Load data
    print(f"\n📊 Loading data for {args.cancer_type}...")
    dataloader = get_feature(args.cancer_type, args.batch_size, training=False)
    print("✅ Data loaded successfully")
    
    # Get predictions
    print(f"\n🔮 Getting predictions...")
    pred_labels_all, embeddings_all = get_predictions(model, dataloader, device)
    print(f"✅ Predictions obtained: {len(pred_labels_all)} samples")
    
    # Filter to only patients with ground truth labels
    print(f"\n🔍 Filtering to patients with ground truth...")
    
    # Load feature file to get patient IDs
    fea_file = f'../subtype_file/fea/{args.cancer_type}/CN.fea'
    fea_data = pd.read_csv(fea_file, header=0, index_col=0, sep=',')
    all_patient_ids = fea_data.columns.tolist()
    
    # Create mapping
    patient_to_idx = {pid: i for i, pid in enumerate(all_patient_ids)}
    gt_patient_ids = gt_df['patient_id'].tolist()
    
    # Get indices of patients with ground truth
    valid_indices = [patient_to_idx[pid] for pid in gt_patient_ids if pid in patient_to_idx]
    
    print(f"   Patients with both predictions and ground truth: {len(valid_indices)}")
    
    # Filter predictions and embeddings
    pred_labels = pred_labels_all[valid_indices]
    embeddings = embeddings_all[valid_indices]
    true_labels = gt_df[label_col].values
    
    print(f"   Final evaluation size: {len(pred_labels)} patients")
    
    # Evaluate
    print(f"\n📈 Computing metrics...")
    metrics = evaluate_clustering(pred_labels, true_labels)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    for metric_name, value in metrics.items():
        print(f"{metric_name:20s}: {value:.4f}")
    print("="*60)
    
    # Create visualizations
    print(f"\n📊 Creating visualizations...")
    plot_confusion_matrix(pred_labels, true_labels, args.cancer_type)
    plot_cluster_distribution(pred_labels, true_labels, args.cancer_type)
    plot_embeddings_tsne(embeddings, true_labels, pred_labels, args.cancer_type)
    
    # Save results
    results_df = pd.DataFrame([metrics])
    results_df['cancer_type'] = args.cancer_type
    results_file = f'results/{args.cancer_type}_evaluation.csv'
    results_df.to_csv(results_file, index=False)
    print(f"\n💾 Saved results to {results_file}")
    
    print("\n✅ Evaluation complete!")