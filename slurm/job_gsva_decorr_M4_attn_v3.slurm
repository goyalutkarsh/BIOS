#!/bin/bash
#SBATCH --job-name=bios_decorr_attn_v3
#SBATCH --output=logs/gsva_decorr_M4_attn_v3_%j.out
#SBATCH --error=logs/gsva_decorr_M4_attn_v3_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=genacc_q

PYBIN=/gpfs/home/ug25b/.conda/envs/bios/bin/python
cd /gpfs/research/fangroup/ug25b/BIOS/Subtype-DCC
export PYTHONPATH=/gpfs/research/fangroup/ug25b/BIOS/Subtype-DCC:$PYTHONPATH
export OMP_NUM_THREADS=8

$PYBIN scripts/train/train_bioanchor.py \
    --head_type       attention \
    --cancer_type     BRCA \
    --batch_size      64 \
    --cluster_number  5 \
    --epochs          400 \
    --lambda_bio      0.1 \
    --bio_dim         4 \
    --bio_anchor_file data/bio_anchors/bio_anchors_BRCA_gsva_decorr_M4.csv \
    --model_path      save/model_bioanchor_gsva_decorr_M4_attn \
    --reload          True \
    --start_epoch     250

echo "Training done at: $(date)"

for epoch in 260 270 280 290 300 320 340 360 380 400; do
    result=$($PYBIN scripts/evaluate/evaluate_clustering.py --cluster_number 5 --cancer_type BRCA --model_path save/model_bioanchor_gsva_decorr_M4_attn/checkpoint_${epoch}.tar 2>/dev/null | grep -E "ARI|V-measure")
    echo "=== epoch $epoch ==="
    echo "$result"
done
