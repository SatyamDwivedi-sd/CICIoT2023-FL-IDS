#!/bin/bash
# =============================================================================
# run_12_fedprox_remaining_multiseed_array.sh — FedProx remaining multi-seed jobs
#
# Already completed:
#   iid seed 123, mu 0.01
#
# Array mapping:
#   task 0: iid          seed 2026
#   task 1: iid          seed 3407
#   task 2: dirichlet_03 seed 123
#   task 3: dirichlet_03 seed 2026
#   task 4: dirichlet_03 seed 3407
#   task 5: dirichlet_01 seed 123
#   task 6: dirichlet_01 seed 2026
#   task 7: dirichlet_01 seed 3407
#
# Submit from project root:
#   cd ~/projects/CICIoT2023 && sbatch slurm/run_12_fedprox_remaining_multiseed_array.sh
# =============================================================================

#SBATCH --job-name=fedprox_ms
#SBATCH --output=experiments/stage2_fl/fedprox/slurm_%A_%a_fedprox.out
#SBATCH --error=experiments/stage2_fl/fedprox/slurm_%A_%a_fedprox.err
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --time=05:00:00
#SBATCH --array=0-7%2

set -euo pipefail

PROJECT="$HOME/projects/CICIoT2023"
DATASETS="$HOME/datasets/CICIoT2023"
PYTHON="$HOME/ciciot_env/bin/python"

PARTITION_MODES=(
    iid
    iid
    dirichlet_03
    dirichlet_03
    dirichlet_03
    dirichlet_01
    dirichlet_01
    dirichlet_01
)

SEEDS=(
    2026
    3407
    123
    2026
    3407
    123
    2026
    3407
)

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
PARTITION_MODE="${PARTITION_MODES[$TASK_ID]}"
SEED="${SEEDS[$TASK_ID]}"

mkdir -p "$PROJECT/experiments/stage2_fl/fedprox" \
         "$PROJECT/experiments/stage2_fl/results" \
         "$PROJECT/experiments/stage2_fl/plots"
cd "$PROJECT"

TASK_LOG="$PROJECT/experiments/stage2_fl/fedprox/slurm_${SLURM_ARRAY_JOB_ID}_${TASK_ID}_${PARTITION_MODE}_seed${SEED}_mu0p01.log"
exec > >(tee -a "$TASK_LOG") 2>&1

echo "=== Job info ==="
echo "Job ID    : $SLURM_JOB_ID"
echo "Array job : $SLURM_ARRAY_JOB_ID"
echo "Task ID   : $TASK_ID"
echo "Node      : $SLURMD_NODENAME"
echo "CPUs      : $SLURM_CPUS_PER_TASK"
echo "GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'not detected')"
echo "Start     : $(date)"
echo "Mode      : $PARTITION_MODE"
echo "Seed      : $SEED"
echo "Mu        : 0.01"
echo "Task log  : $TASK_LOG"
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

PARTITION_FILE="$PROJECT/experiments/stage2_fl/partitioning/${PARTITION_MODE}_client_indices.npz"
if [ ! -f "$PARTITION_FILE" ]; then
    echo "ERROR: $PARTITION_FILE not found — run 09_FL_Partitioning.ipynb first."
    exit 1
fi
echo "  OK  ${PARTITION_MODE}_client_indices.npz"
echo ""

echo "=== Running 12_fedprox.py (${PARTITION_MODE}, seed ${SEED}, mu 0.01) ==="
$PYTHON scripts/12_fedprox.py \
    --partition-mode "$PARTITION_MODE" \
    --seed "$SEED" \
    --mu 0.01 \
    --data-dir "$DATASETS"

echo ""
echo "=== Done: $(date) ==="
echo "Outputs:"
ls -lh "$PROJECT/experiments/stage2_fl/fedprox/" 2>/dev/null
ls -lh "$PROJECT/experiments/stage2_fl/results/" 2>/dev/null
ls -lh "$PROJECT/experiments/stage2_fl/plots/"   2>/dev/null
