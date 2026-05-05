#!/bin/bash
# =============================================================================
# run_12_fedprox_smoke_iid_seed123.sh — FedProx smoke test, IID seed 123
#
# Runs one Stage 2B FedProx smoke test:
#   partition mode: iid
#   seed:           123
#   mu:             0.01
#
# Submit from project root:
#   cd ~/projects/CICIoT2023 && sbatch slurm/run_12_fedprox_smoke_iid_seed123.sh
# =============================================================================

#SBATCH --job-name=fedprox_smoke
#SBATCH --output=experiments/stage2_fl/fedprox/slurm_%j_iid_seed123_mu0p01.out
#SBATCH --error=experiments/stage2_fl/fedprox/slurm_%j_iid_seed123_mu0p01.err
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --time=03:00:00

set -euo pipefail

PROJECT="$HOME/projects/CICIoT2023"
DATASETS="$HOME/datasets/CICIoT2023"
PYTHON="$HOME/ciciot_env/bin/python"

mkdir -p "$PROJECT/experiments/stage2_fl/fedprox" \
         "$PROJECT/experiments/stage2_fl/results" \
         "$PROJECT/experiments/stage2_fl/plots"
cd "$PROJECT"

echo "=== Job info ==="
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURMD_NODENAME"
echo "CPUs      : $SLURM_CPUS_PER_TASK"
echo "GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'not detected')"
echo "Start     : $(date)"
echo "Mode      : iid"
echo "Seed      : 123"
echo "Mu        : 0.01"
echo ""

module load miniconda/miniconda3

# ── Prerequisite checks: Stage 1 cat8 files ───────────────────────────────────
echo "=== Checking prerequisites ==="
for f in X_train_cat8.csv y_train_cat8.csv \
          X_val_cat8.csv   y_val_cat8.csv   \
          X_test_cat8.csv  y_test_cat8.csv  \
          cat8_mapping.json; do
    if [ ! -f "$DATASETS/$f" ]; then
        echo "ERROR: $DATASETS/$f not found — run Stage 1 pipeline first."
        exit 1
    fi
    echo "  OK  $f"
done

PARTITION_FILE="$PROJECT/experiments/stage2_fl/partitioning/iid_client_indices.npz"
if [ ! -f "$PARTITION_FILE" ]; then
    echo "ERROR: $PARTITION_FILE not found — run 09_FL_Partitioning.ipynb first."
    exit 1
fi
echo "  OK  iid_client_indices.npz"
echo ""

echo "=== Running 12_fedprox.py (iid, seed 123, mu 0.01) ==="
$PYTHON scripts/12_fedprox.py \
    --partition-mode iid \
    --seed 123 \
    --mu 0.01 \
    --data-dir "$DATASETS"

echo ""
echo "=== Done: $(date) ==="
echo "Outputs:"
ls -lh "$PROJECT/experiments/stage2_fl/fedprox/" 2>/dev/null
ls -lh "$PROJECT/experiments/stage2_fl/results/" 2>/dev/null
ls -lh "$PROJECT/experiments/stage2_fl/plots/"   2>/dev/null
