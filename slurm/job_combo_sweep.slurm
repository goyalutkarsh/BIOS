#!/bin/bash
#SBATCH --job-name=bios_combo_sweep
#SBATCH --output=logs/combo_sweep_%j.out
#SBATCH --error=logs/combo_sweep_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=backfill2
#SBATCH --gres=gpu:ada4500:1

PYBIN=/gpfs/home/ug25b/.conda/envs/bios/bin/python
DIR=/gpfs/research/fangroup/ug25b/BIOS/Subtype-DCC

cd $DIR

run_combo() {
    NAME=$1
    FILE=$2
    DIM=$3

    echo "=============================="
    echo "Running: $NAME | dim=$DIM"
    echo "=============================="

    $PYBIN train_bioanchor_mlp.py \
        --bio_dim $DIM \
        --bio_anchor_file $FILE \
        --model_path save/model_combo_mlp_${NAME} \
        --epochs 200 \
        --lambda_bio 0.1

    echo "--- MLP $NAME RESULTS ---"
    LAST=$(ls -t save/model_combo_mlp_${NAME}/*.pth 2>/dev/null | head -1)
    $PYBIN scripts/evaluate/evaluate_clustering.py --cancer_type BRCA --model_path $LAST

    $PYBIN train_bioanchor_linear.py \
        --bio_dim $DIM \
        --bio_anchor_file $FILE \
        --model_path save/model_combo_linear_${NAME} \
        --epochs 200 \
        --lambda_bio 0.1

    echo "--- Linear $NAME RESULTS ---"
    LAST=$(ls -t save/model_combo_linear_${NAME}/*.pth 2>/dev/null | head -1)
    $PYBIN scripts/evaluate/evaluate_clustering.py --cancer_type BRCA --model_path $LAST
}

run_combo "3base"            "data/bio_anchors/bio_anchors_BRCA_combo_3base.csv"                  3
run_combo "3plus_PI3K"       "data/bio_anchors/bio_anchors_BRCA_combo_3plus_PI3K_score.csv"       4
run_combo "3plus_p53"        "data/bio_anchors/bio_anchors_BRCA_combo_3plus_p53_score.csv"        4
run_combo "3plus_MYC"        "data/bio_anchors/bio_anchors_BRCA_combo_3plus_MYC_score.csv"        4
run_combo "3plus_MAPK"       "data/bio_anchors/bio_anchors_BRCA_combo_3plus_MAPK_score.csv"       4
run_combo "3plus_WNT"        "data/bio_anchors/bio_anchors_BRCA_combo_3plus_WNT_score.csv"        4
run_combo "3plus_hypoxia"    "data/bio_anchors/bio_anchors_BRCA_combo_3plus_hypoxia_score.csv"    4
run_combo "3plus_cellcycle"  "data/bio_anchors/bio_anchors_BRCA_combo_3plus_cellcycle_score.csv"  4
run_combo "3plus_apoptosis"  "data/bio_anchors/bio_anchors_BRCA_combo_3plus_apoptosis_score.csv"  4
run_combo "3plus_angio"      "data/bio_anchors/bio_anchors_BRCA_combo_3plus_angio_score.csv"      4
run_combo "3plus_dna_repair" "data/bio_anchors/bio_anchors_BRCA_combo_3plus_dna_repair_score.csv" 4
run_combo "3plus_TMB"        "data/bio_anchors/bio_anchors_BRCA_combo_3plus_TMB_score.csv"        4
run_combo "14all"            "data/bio_anchors/bio_anchors_BRCA_combo_14all.csv"                  14

echo "=============================="
echo "ALL COMBOS COMPLETE"
echo "=============================="