#!/bin/bash
# =============================================================================
# run_05_binary_dl.sh — Binary DL baseline (MLP + 1D CNN, PyTorch)
#
# Produces: models/binary/mlp_best.pt
#           models/binary/cnn_best.pt
#           results/binary/dl_baseline_results_binary.csv
#
# NOTE: Binary DL results already exist locally. Only run this if you need
#       to regenerate or verify results on the cluster.
#
# Submit:
#   sbatch slurm/run_05_binary_dl.sh
# =============================================================================

#SBATCH --job-name=ciciot_dl_binary
#SBATCH --output=logs/dl_binary_%j.out
#SBATCH --error=logs/dl_binary_%j.err
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

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
echo "GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'not detected')"
echo "Start     : $(date)"
echo ""

module load miniconda/miniconda3

echo "=== Running 05_train_binary_dl.py ==="
$PYTHON scripts/05_train_binary_dl.py \
    --data-dir    "$DATASETS" \
    --models-dir  "$PROJECT/models/binary" \
    --results-dir "$PROJECT/results/binary" \
    --epochs      10 \
    --batch-size  4096 \
    --lr          1e-4 \
    --dropout     0.3

echo ""
echo "=== Done: $(date) ==="
ls -lh "$PROJECT/results/binary/" 2>/dev/null
