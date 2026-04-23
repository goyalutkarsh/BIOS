#!/bin/bash
#SBATCH --job-name=bioanchor_3dim
#SBATCH --output=logs/bioanchor_3dim_%j.out
#SBATCH --error=logs/bioanchor_3dim_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=genacc_q

PYBIN=/gpfs/home/ug25b/.conda/envs/bios/bin/python

mkdir -p logs save/model_bioanchor_3dim results

echo "=== Training Bio-Anchor 3D (No Placeholders) ==="
date

# Train with 3D bio-anchors
$PYBIN train_bioanchor_3dim.py \
    --cancer_type BRCA \
    --batch_size 32 \
    --cluster_number 5 \
    --epochs 200 \
    --model_path save/model_bioanchor_3dim

# Evaluate
$PYBIN scripts/evaluate/evaluate_clustering.py \
    --cancer_type BRCA \
    --model_path save/model_bioanchor_3dim/checkpoint_190.tar

# Rename results
mv results/BRCA_evaluation.csv results/BRCA_3dim_evaluation.csv
mv results/BRCA_confusion_matrix.png results/BRCA_3dim_confusion.png
mv results/BRCA_cluster_distribution.png results/BRCA_3dim_distribution.png
mv results/BRCA_tsne.png results/BRCA_3dim_tsne.png

echo "=== Completed 3D Training ==="
date