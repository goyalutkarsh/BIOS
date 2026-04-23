#!/bin/bash
#SBATCH --job-name=bios_combo_1
#SBATCH --output=logs/combo_sweep_1_%j.out
#SBATCH --error=logs/combo_sweep_1_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=genacc_q

PYBIN=/gpfs/home/ug25b/.conda/envs/bios/bin/python
DIR=/gpfs/research/fangroup/ug25b/BIOS/Subtype-DCC

cd $DIR
mkdir -p logs

run_combo() {
    NAME=$1
    FILE=$2
    DIM=$3

    echo "=============================="
    echo "Running: $NAME | dim=$DIM"
    echo "=============================="

    mkdir -p save/model_combo_mlp_${NAME} save/model_combo_linear_${NAME}

    $PYBIN train_bioanchor_mlp.py \
        --cancer_type BRCA \
        --batch_size 64 \
        --cluster_number 5 \
        --epochs 200 \
        --lambda_bio 0.1 \
        --bio_dim $DIM \
        --bio_anchor_file $FILE \
        --model_path save/model_combo_mlp_${NAME}

    echo "--- MLP $NAME RESULTS ---"
    LAST=$(ls -t save/model_combo_mlp_${NAME}/checkpoint_*.tar 2>/dev/null | head -1)
    $PYBIN scripts/evaluate/evaluate_clustering.py --cancer_type BRCA --model_path $LAST

    $PYBIN train_bioanchor_linear.py \
        --cancer_type BRCA \
        --batch_size 64 \
        --cluster_number 5 \
        --epochs 200 \
        --lambda_bio 0.1 \
        --bio_dim $DIM \
        --bio_anchor_file $FILE \
        --model_path save/model_combo_linear_${NAME}

    echo "--- Linear $NAME RESULTS ---"
    LAST=$(ls -t save/model_combo_linear_${NAME}/checkpoint_*.tar 2>/dev/null | head -1)
    $PYBIN scripts/evaluate/evaluate_clustering.py --cancer_type BRCA --model_path $LAST
}

run_combo "3base"            "data/bio_anchors/bio_anchors_BRCA_combo_3base.csv"                  3
run_combo "3plus_PI3K"       "data/bio_anchors/bio_anchors_BRCA_combo_3plus_PI3K_score.csv"       4
run_combo "3plus_p53"        "data/bio_anchors/bio_anchors_BRCA_combo_3plus_p53_score.csv"        4
run_combo "3plus_MYC"        "data/bio_anchors/bio_anchors_BRCA_combo_3plus_MYC_score.csv"        4
run_combo "3plus_MAPK"       "data/bio_anchors/bio_anchors_BRCA_combo_3plus_MAPK_score.csv"       4
run_combo "3plus_WNT"        "data/bio_anchors/bio_anchors_BRCA_combo_3plus_WNT_score.csv"        4
run_combo "3plus_hypoxia"    "data/bio_anchors/bio_anchors_BRCA_combo_3plus_hypoxia_score.csv"    4

echo "=============================="
echo "SWEEP 1 COMPLETE"
echo "=============================="