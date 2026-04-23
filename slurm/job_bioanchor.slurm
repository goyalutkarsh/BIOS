#!/bin/bash
#SBATCH --job-name=bioanchor_brca
#SBATCH --output=logs/bioanchor_%j.out
#SBATCH --error=logs/baseline_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=genacc_q

# Set Python binary explicitly
PYBIN=/gpfs/home/ug25b/.conda/envs/bios/bin/python

# Create directories
mkdir -p logs results save/model_bioanchor

# Print job info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Set number of threads
export OMP_NUM_THREADS=8

# Run baseline training
$PYBIN train_bioanchor.py \
    --cancer_type BRCA \
    --batch_size 32 \
    --cluster_number 5 \
    --epochs 600 \
    --model_path save/model_bioanchor

echo "Training completed at: $(date)"

# Run evaluation
LAST_CHECKPOINT=$(ls -t save/model_bioanchor/checkpoint_*.tar | head -1)
$PYBIN scripts/evaluate/evaluate_clustering.py --cancer_type BRCA --model_path $LAST_CHECKPOINT

# Rename results
mv results/BRCA_evaluation.csv results/BRCA_bioanchor_evaluation.csv
mv results/BRCA_confusion_matrix.png results/BRCA_bioanchor_confusion_matrix.png
mv results/BRCA_cluster_distribution.png results/BRCA_bioanchor_cluster_distribution.png
mv results/BRCA_tsne.png results/BRCA_bioanchor_tsne.png

echo "Job finished at: $(date)"
