#!/bin/bash
#SBATCH --job-name=bios_attention
#SBATCH --output=logs/attention_%j.out
#SBATCH --error=logs/attention_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=genacc_q

PYBIN=/gpfs/home/ug25b/.conda/envs/bios/bin/python

mkdir -p logs save/model_bioanchor_attention_14dim_200ep results

echo "=== Bio-Anchor Head: attention ==="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

cd /gpfs/research/fangroup/ug25b/BIOS/Subtype-DCC
export PYTHONPATH=/gpfs/research/fangroup/ug25b/BIOS/Subtype-DCC:$PYTHONPATH
export OMP_NUM_THREADS=8

# OLD: called train_bioanchor_linear.py or train_bioanchor_mlp.py separately
# NEW: single unified script, head selected via --head_type
$PYBIN scripts/train/train_bioanchor.py \
    --head_type       attention \
    --cancer_type     BRCA \
    --batch_size      64 \
    --cluster_number  5 \
    --epochs          600 \
    --lambda_bio      0.1 \
    --bio_dim         14 \
    --bio_anchor_file data/bio_anchors/bio_anchors_BRCA_14dim.csv \
    --model_path      save/model_bioanchor_attention_14dim_200ep

echo "Training completed at: $(date)"

# evaluate on best checkpoint — same as your old scripts
LAST_CHECKPOINT=$(ls -t save/model_bioanchor_attention_14dim_200ep/checkpoint_*.tar | head -1)
$PYBIN scripts/evaluate/evaluate_clustering.py \
    --cluster_number  5 \
    --cancer_type BRCA \
    --model_path $LAST_CHECKPOINT

# rename results with head_type prefix so all three coexist in results/
# OLD: hardcoded _linear_ or _mlp_ prefix per script
# NEW: attention prefix — works for all three
mv results/BRCA_evaluation.csv          results/BRCA_attention_evaluation.csv
mv results/BRCA_confusion_matrix.png    results/BRCA_attention_confusion.png
mv results/BRCA_cluster_distribution.png results/BRCA_attention_distribution.png
mv results/BRCA_tsne.png               results/BRCA_attention_tsne.png

echo "=== attention complete ==="
echo "Job finished at: $(date)"
