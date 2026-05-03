#!/bin/bash
# =============================================================================
# run_08_fedavg_multiseed_array.sh — FedAvg multi-seed SLURM array (Stage 2)
#
# Runs each partition mode with each additional seed:
#   partition modes: iid, dirichlet_03, dirichlet_01
#   seeds:           123, 2026, 3407
#
# Array mapping:
#   task 0: iid          seed 123
#   task 1: iid          seed 2026
#   task 2: iid          seed 3407
#   task 3: dirichlet_03 seed 123
#   task 4: dirichlet_03 seed 2026
#   task 5: dirichlet_03 seed 3407
#   task 6: dirichlet_01 seed 123
#   task 7: dirichlet_01 seed 2026
#   task 8: dirichlet_01 seed 3407
#
# Submit from project root:
#   cd ~/projects/CICIoT2023 && sbatch slurm/run_08_fedavg_multiseed_array.sh
# =============================================================================

#SBATCH --job-name=fedavg_ms
#SBATCH --output=experiments/stage2_fl/fedavg/slurm_%A_%a_multiseed.out
#SBATCH --error=experiments/stage2_fl/fedavg/slurm_%A_%a_multiseed.err
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --time=03:00:00
#SBATCH --array=0-8

set -euo pipefail

PROJECT="$HOME/projects/CICIoT2023"
DATASETS="$HOME/datasets/CICIoT2023"
PYTHON="$HOME/ciciot_env/bin/python"

PARTITION_MODES=(iid dirichlet_03 dirichlet_01)
SEEDS=(123 2026 3407)

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
MODE_INDEX=$((TASK_ID / ${#SEEDS[@]}))
SEED_INDEX=$((TASK_ID % ${#SEEDS[@]}))

PARTITION_MODE="${PARTITION_MODES[$MODE_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"

mkdir -p "$PROJECT/experiments/stage2_fl/fedavg" \
         "$PROJECT/experiments/stage2_fl/results" \
         "$PROJECT/experiments/stage2_fl/plots"
cd "$PROJECT"

TASK_LOG="$PROJECT/experiments/stage2_fl/fedavg/slurm_${SLURM_ARRAY_JOB_ID}_${TASK_ID}_${PARTITION_MODE}_seed${SEED}.log"
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

echo "=== Running 08_fedavg.py (${PARTITION_MODE}, seed ${SEED}) ==="
$PYTHON scripts/08_fedavg.py \
    --partition-mode "$PARTITION_MODE" \
    --seed "$SEED" \
    --data-dir "$DATASETS"

echo ""
echo "=== Done: $(date) ==="
echo "Outputs:"
ls -lh "$PROJECT/experiments/stage2_fl/fedavg/"  2>/dev/null
ls -lh "$PROJECT/experiments/stage2_fl/results/" 2>/dev/null
ls -lh "$PROJECT/experiments/stage2_fl/plots/"   2>/dev/null
