#!/bin/bash
#SBATCH --job-name=lambda_test
#SBATCH --output=logs/lambda_%A_%a.out
#SBATCH --error=logs/lambda_%A_%a.err
#SBATCH --array=0-2
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=genacc_q

PYBIN=/gpfs/home/ug25b/.conda/envs/bios/bin/python

# Test 3 lambda values
LAMBDAS=(0.1 0.2 0.3)
LAMBDA=${LAMBDAS[$SLURM_ARRAY_TASK_ID]}

mkdir -p logs save/model_bioanchor_lambda${LAMBDA} results

echo "=== Testing lambda_bio = $LAMBDA ==="
date

# Train
$PYBIN train_bioanchor.py \
    --cancer_type BRCA \
    --batch_size 32 \
    --cluster_number 5 \
    --epochs 200 \
    --lambda_bio $LAMBDA \
    --model_path save/model_bioanchor_lambda${LAMBDA}

# Evaluate
$PYBIN scripts/evaluate/evaluate_clustering.py \
    --cancer_type BRCA \
    --model_path save/model_bioanchor_lambda${LAMBDA}/checkpoint_190.tar

# Rename results to preserve them
mv results/BRCA_evaluation.csv results/BRCA_lambda${LAMBDA}_evaluation.csv
mv results/BRCA_confusion_matrix.png results/BRCA_lambda${LAMBDA}_confusion.png
mv results/BRCA_cluster_distribution.png results/BRCA_lambda${LAMBDA}_distribution.png
mv results/BRCA_tsne.png results/BRCA_lambda${LAMBDA}_tsne.png

echo "=== Completed lambda=$LAMBDA ==="
date