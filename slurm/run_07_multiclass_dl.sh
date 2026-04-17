#!/bin/bash
# =============================================================================
# run_07_multiclass_dl.sh — Multiclass DL baseline (MLP + 1D CNN, TF/Keras)
#
# Requires: extracted/X_train_balanced_cat8.csv (run run_03_balance.sh first)
# Produces: models/cat8/mlp_cat8_best.keras
#           models/cat8/cnn_cat8_best.keras
#           results/cat8/results_dl_baseline_cat8.csv
#           results/cat8/experiment_log_multiclass_dl.json
#
# Submit after balance job completes:
#   sbatch slurm/run_07_multiclass_dl.sh
#
# Or chain automatically:
#   BALANCE_JOB=$(sbatch --parsable slurm/run_03_balance.sh)
#   sbatch --dependency=afterok:$BALANCE_JOB slurm/run_07_multiclass_dl.sh
# =============================================================================

#SBATCH --job-name=ciciot_dl_cat8
#SBATCH --output=logs/dl_cat8_%j.out
#SBATCH --error=logs/dl_cat8_%j.err
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8           # TF data pipeline workers
#SBATCH --time=02:00:00             # MLP + CNN, 15 epochs each — ~30 min on RTX 5070 Ti

set -euo pipefail

PROJECT="$HOME/projects/CICIoT2023"
DATASETS="$HOME/datasets/CICIoT2023"
PYTHON="$HOME/ciciot_env/bin/python"
LOGS="$PROJECT/logs"

mkdir -p "$LOGS" "$PROJECT/models/cat8" "$PROJECT/results/cat8"
cd "$PROJECT"

echo "=== Job info ==="
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURMD_NODENAME"
echo "CPUs      : $SLURM_CPUS_PER_TASK"
echo "GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'not detected')"
echo "Start     : $(date)"
echo ""

module load miniconda/miniconda3

# Verify balanced training file exists
if [ ! -f "$DATASETS/X_train_balanced_cat8.csv" ]; then
    echo "ERROR: X_train_balanced_cat8.csv not found — run run_03_balance.sh first."
    exit 1
fi

echo "=== Running 07_train_multiclass_dl.py ==="
$PYTHON scripts/07_train_multiclass_dl.py \
    --data-dir    "$DATASETS" \
    --models-dir  "$PROJECT/models/cat8" \
    --results-dir "$PROJECT/results/cat8" \
    --epochs      15 \
    --batch-size  4096 \
    --lr          1e-4 \
    --patience    3 \
    --dropout     0.3
# Default run: no --use-class-weights (matches notebook behaviour)

echo ""
echo "=== Done: $(date) ==="
echo "Results:"
ls -lh "$PROJECT/results/cat8/" 2>/dev/null
