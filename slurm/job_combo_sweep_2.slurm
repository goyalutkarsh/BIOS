#!/bin/bash
#SBATCH --job-name=bios_combo_2
#SBATCH --output=logs/combo_sweep_2_%j.out
#SBATCH --error=logs/combo_sweep_2_%j.err
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

run_combo "3plus_cellcycle"  "data/bio_anchors/bio_anchors_BRCA_combo_3plus_cellcycle_score.csv"  4
run_combo "3plus_apoptosis"  "data/bio_anchors/bio_anchors_BRCA_combo_3plus_apoptosis_score.csv"  4
run_combo "3plus_angio"      "data/bio_anchors/bio_anchors_BRCA_combo_3plus_angio_score.csv"      4
run_combo "3plus_dna_repair" "data/bio_anchors/bio_anchors_BRCA_combo_3plus_dna_repair_score.csv" 4
run_combo "3plus_TMB"        "data/bio_anchors/bio_anchors_BRCA_combo_3plus_TMB_score.csv"        4
run_combo "14all"            "data/bio_anchors/bio_anchors_BRCA_combo_14all.csv"                  14

echo "=============================="
echo "SWEEP 2 COMPLETE"
echo "=============================="