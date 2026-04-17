#!/bin/bash
# =============================================================================
# run_04_binary_ml.sh — Binary ML baseline (RF + XGBoost)
#
# Produces: models/binary/rf_model.pkl
#           models/binary/xgb_model.pkl
#           results/binary/ml_baseline_results_binary.csv
#
# NOTE: Binary results already exist locally. Only run this if you need
#       to regenerate or verify results on the cluster.
#
# Submit:
#   sbatch slurm/run_04_binary_ml.sh
# =============================================================================

#SBATCH --job-name=ciciot_ml_binary
#SBATCH --output=logs/ml_binary_%j.out
#SBATCH --error=logs/ml_binary_%j.err
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18          # RF on 16M rows — needs all CPUs
#SBATCH --time=04:00:00

set -euo pipefail

PROJECT="$HOME/projects/CICIoT2023"
DATASETS="$HOME/datasets/CICIoT2023"
PYTHON="$HOME/ciciot_env/bin/python"
LOGS="$PROJECT/logs"

mkdir -p "$LOGS" "$PROJECT/models/binary" "$PROJECT/results/binary"
cd "$PROJECT"

echo "=== Job info ==="
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURMD_NODENAME"
echo "CPUs      : $SLURM_CPUS_PER_TASK"
echo "Start     : $(date)"
echo ""

module load miniconda/miniconda3

echo "=== Running 04_train_binary_ml.py ==="
$PYTHON scripts/04_train_binary_ml.py \
    --data-dir    "$DATASETS" \
    --models-dir  "$PROJECT/models/binary" \
    --results-dir "$PROJECT/results/binary"

echo ""
echo "=== Done: $(date) ==="
ls -lh "$PROJECT/results/binary/" 2>/dev/null
