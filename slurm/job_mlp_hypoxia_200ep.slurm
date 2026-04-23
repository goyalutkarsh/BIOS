#!/bin/bash
#SBATCH --job-name=bios_hypoxia_mlp
#SBATCH --output=logs/hypoxia_mlp_%j.out
#SBATCH --error=logs/hypoxia_mlp_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=genacc_q

PYBIN=/gpfs/home/ug25b/.conda/envs/bios/bin/python
DIR=/gpfs/research/fangroup/ug25b/BIOS/Subtype-DCC

cd $DIR
mkdir -p logs save/model_combo_mlp_3plus_hypoxia results

echo "=== 3+Hypoxia, MLP, 200 epochs ==="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

export OMP_NUM_THREADS=8

$PYBIN train_bioanchor_mlp.py \
    --cancer_type BRCA \
    --batch_size 64 \
    --cluster_number 5 \
    --epochs 200 \
    --lambda_bio 0.1 \
    --bio_dim 4 \
    --bio_hidden_dim 64 \
    --bio_anchor_file data/bio_anchors/bio_anchors_BRCA_combo_3plus_hypoxia_score.csv \
    --model_path save/model_combo_mlp_3plus_hypoxia

echo "Training completed at: $(date)"

LAST_CHECKPOINT=$(ls -t save/model_combo_mlp_3plus_hypoxia/checkpoint_*.tar | head -1)
$PYBIN scripts/evaluate/evaluate_clustering.py --cancer_type BRCA --model_path $LAST_CHECKPOINT

mv results/BRCA_evaluation.csv           results/BRCA_hypoxia_mlp_evaluation.csv
mv results/BRCA_confusion_matrix.png     results/BRCA_hypoxia_mlp_confusion.png
mv results/BRCA_cluster_distribution.png results/BRCA_hypoxia_mlp_distribution.png
mv results/BRCA_tsne.png                 results/BRCA_hypoxia_mlp_tsne.png

echo "=== Hypoxia MLP 200ep Complete ==="
echo "Job finished at: $(date)"
