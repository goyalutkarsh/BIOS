#!/bin/bash
#SBATCH --job-name=bios_gsva_mlp
#SBATCH --output=logs/gsva_mlp_%j.out
#SBATCH --error=logs/gsva_mlp_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=genacc_q

PYBIN=/gpfs/home/ug25b/.conda/envs/bios/bin/python
mkdir -p logs save/model_bioanchor_gsva_mlp_M8 results

echo "=== Bio-Anchor Head: mlp | GSVA M8 anchors ==="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

cd /gpfs/research/fangroup/ug25b/BIOS/Subtype-DCC
export PYTHONPATH=/gpfs/research/fangroup/ug25b/BIOS/Subtype-DCC:$PYTHONPATH
export OMP_NUM_THREADS=8

echo "Python: $PYBIN"
echo "PWD: $(pwd)"
ls data/bio_anchors/bio_anchors_BRCA_gsva_M8.csv

$PYBIN scripts/train/train_bioanchor.py \
    --head_type       mlp \
    --cancer_type     BRCA \
    --batch_size      64 \
    --cluster_number  5 \
    --epochs          600 \
    --lambda_bio      0.1 \
    --bio_dim         8 \
    --bio_anchor_file data/bio_anchors/bio_anchors_BRCA_gsva_M8.csv \
    --model_path      save/model_bioanchor_gsva_mlp_M8

echo "Training completed at: $(date)"

# LAST_CHECKPOINT=$(ls -t save/model_bioanchor_gsva_mlp_M8/checkpoint_*.tar | head -1)
LAST_CHECKPOINT=save/model_bioanchor_gsva_linear_M8/best_checkpoint.tar
$PYBIN scripts/evaluate/evaluate_clustering.py \
    --cluster_number  5 \
    --cancer_type     BRCA \
    --model_path      $LAST_CHECKPOINT

mv results/BRCA_evaluation.csv           results/BRCA_gsva_mlp_M8_evaluation.csv
mv results/BRCA_confusion_matrix.png     results/BRCA_gsva_mlp_M8_confusion.png     2>/dev/null || true
mv results/BRCA_cluster_distribution.png results/BRCA_gsva_mlp_M8_distribution.png  2>/dev/null || true
mv results/BRCA_tsne.png                 results/BRCA_gsva_mlp_M8_tsne.png           2>/dev/null || true

echo "=== gsva_mlp_M8 complete ==="
echo "Job finished at: $(date)"
