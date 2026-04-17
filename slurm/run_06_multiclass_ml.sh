#!/bin/bash
# =============================================================================
# run_06_multiclass_ml.sh — Multiclass ML baseline (RF + XGBoost, 8-category)
#
# Requires: extracted/X_train_balanced_cat8.csv (run run_03_balance.sh first)
# Produces: models/cat8/rf_model_cat8.pkl
#           models/cat8/xgb_model_cat8.pkl
#           results/cat8/results_ml_baseline_cat8.csv
#           results/cat8/experiment_log_multiclass_ml.json
#
# Submit after balance job completes:
#   sbatch slurm/run_06_multiclass_ml.sh
#
# Or chain automatically:
#   BALANCE_JOB=$(sbatch --parsable slurm/run_03_balance.sh)
#   sbatch --dependency=afterok:$BALANCE_JOB slurm/run_06_multiclass_ml.sh
# =============================================================================

#SBATCH --job-name=ciciot_ml_cat8
#SBATCH --output=logs/ml_cat8_%j.out
#SBATCH --error=logs/ml_cat8_%j.err
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18          # RF uses n_jobs=-1 → benefits from all cores
#SBATCH --time=04:00:00             # RF on 1.6M rows + XGBoost GPU

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
echo "Start     : $(date)"
echo ""

module load miniconda/miniconda3

# Verify balanced training file exists
if [ ! -f "$DATASETS/X_train_balanced_cat8.csv" ]; then
    echo "ERROR: X_train_balanced_cat8.csv not found — run run_03_balance.sh first."
    exit 1
fi

echo "=== Running 06_train_multiclass_ml.py ==="
$PYTHON scripts/06_train_multiclass_ml.py \
    --data-dir    "$DATASETS" \
    --models-dir  "$PROJECT/models/cat8" \
    --results-dir "$PROJECT/results/cat8" \
    --rf-estimators  200 \
    --xgb-estimators 300 \
    --xgb-max-depth  8
# Note: XGBoost will use GPU automatically (no --no-gpu flag)

echo ""
echo "=== Done: $(date) ==="
echo "Results:"
ls -lh "$PROJECT/results/cat8/" 2>/dev/null
