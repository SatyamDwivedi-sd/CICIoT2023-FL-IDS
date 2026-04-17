#!/bin/bash
# =============================================================================
# run_03_balance.sh — SMOTE-ENN class balancing for cat8 training split
#
# MUST run before run_06_multiclass_ml.sh and run_07_multiclass_dl.sh.
# Produces: extracted/X_train_balanced_cat8.csv
#           extracted/y_train_balanced_cat8.csv
#
# Submit from project root:
#   sbatch slurm/run_03_balance.sh
# =============================================================================

#SBATCH --job-name=ciciot_balance
#SBATCH --output=logs/balance_%j.out
#SBATCH --error=logs/balance_%j.err
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18          # SMOTE-ENN parallelises over CPUs
#SBATCH --time=06:00:00             # SMOTE-ENN on 16M rows — allow 6h

set -euo pipefail

PROJECT="$HOME/projects/CICIoT2023"
DATASETS="$HOME/datasets/CICIoT2023"
PYTHON="$HOME/ciciot_env/bin/python"
LOGS="$PROJECT/logs"

mkdir -p "$LOGS"
cd "$PROJECT"

echo "=== Job info ==="
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURMD_NODENAME"
echo "CPUs      : $SLURM_CPUS_PER_TASK"
echo "Start     : $(date)"
echo ""

module load miniconda/miniconda3

echo "=== Running 03_balance.py ==="
$PYTHON scripts/03_balance.py \
    --data-dir "$DATASETS" \
    --out-dir  "$DATASETS" \
    --n-jobs   "$SLURM_CPUS_PER_TASK"

echo ""
echo "=== Done: $(date) ==="
echo "Output files:"
ls -lh "$DATASETS/X_train_balanced_cat8.csv" \
       "$DATASETS/y_train_balanced_cat8.csv" 2>/dev/null || echo "WARNING: output files not found"
