#!/bin/bash
#SBATCH --job-name=bios_repro_old
#SBATCH --output=logs/repro_linear_old_%j.out
#SBATCH --error=logs/repro_linear_old_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=genacc_q

PYBIN=/gpfs/home/ug25b/.conda/envs/bios/bin/python

mkdir -p logs save/model_repro_linear_old results

echo "=== Reproduction run: old linear architecture ==="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

cd /gpfs/research/fangroup/ug25b/BIOS/Subtype-DCC
export PYTHONPATH=/gpfs/research/fangroup/ug25b/BIOS/Subtype-DCC:$PYTHONPATH
export OMP_NUM_THREADS=8

# exact same config as the original run that gave V=0.4855
$PYBIN scripts/train/train_bioanchor_linear_OLD.py \
    --cancer_type     BRCA \
    --batch_size      64 \
    --cluster_number  5 \
    --epochs          200 \
    --lambda_bio      0.1 \
    --bio_dim         4 \
    --bio_anchor_file data/bio_anchors/bio_anchors_BRCA_combo_3plus_apoptosis_score.csv \
    --model_path      save/model_repro_linear_old

echo "Training completed at: $(date)"

LAST_CHECKPOINT=$(ls -t save/model_repro_linear_old/checkpoint_*.tar | head -1)
$PYBIN scripts/evaluate/evaluate_clustering.py \
    --cluster_number  5 \
    --cancer_type     BRCA \
    --model_path      $LAST_CHECKPOINT

mv results/BRCA_evaluation.csv           results/BRCA_repro_linear_old_evaluation.csv
mv results/BRCA_confusion_matrix.png     results/BRCA_repro_linear_old_confusion.png
mv results/BRCA_cluster_distribution.png results/BRCA_repro_linear_old_distribution.png
mv results/BRCA_tsne.png                results/BRCA_repro_linear_old_tsne.png

echo "=== Reproduction run complete ==="
echo "Job finished at: $(date)"
