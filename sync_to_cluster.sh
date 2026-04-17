#!/bin/bash
# =============================================================================
# sync_to_cluster.sh — Sync code and preprocessed data to the cluster
#
# Usage:
#   bash sync_to_cluster.sh <user>@<login-node>
#
# Example:
#   bash sync_to_cluster.sh satyam@cyberlab-login.university.ac.za
#
# What this transfers:
#   FAST (code + config):   src/, scripts/, slurm/, pyproject.toml,
#                           requirements.txt, extracted/*.pkl, extracted/*.json
#   SLOW (data, ~14 GB):    extracted/X_*.csv, extracted/y_*.csv
#
# What is intentionally EXCLUDED:
#   extracted/MERGED_CSV/       — raw data, not needed (already preprocessed)
#   extracted/ciciot_clean.csv  — 4.7 GB, not needed (splits already exist)
#   extracted/legacy_old/       — archived, not needed
#   eda_outputs/                — notebooks, not for cluster
#   raw_downloads/              — original zips, not for cluster
#   models/                     — will be created by training jobs
#   results/                    — will be created by training jobs
# =============================================================================

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: bash sync_to_cluster.sh <user>@<login-node>"
    exit 1
fi

REMOTE="$1"
REMOTE_DIR="~/CICIoT2023"
LOCAL="$(cd "$(dirname "$0")" && pwd)"

echo "=== Syncing to $REMOTE:$REMOTE_DIR ==="
echo ""

# ── 1. Create remote directory structure ────────────────────────────────────
echo "[1/4] Creating remote directories..."
ssh "$REMOTE" "mkdir -p $REMOTE_DIR/extracted $REMOTE_DIR/logs $REMOTE_DIR/models $REMOTE_DIR/results"

# ── 2. Sync code (fast — MB range) ──────────────────────────────────────────
echo "[2/4] Syncing code and config..."
rsync -avz --progress \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.egg-info' \
    --exclude='.git' \
    --exclude='eda_outputs/' \
    --exclude='raw_downloads/' \
    --exclude='models/' \
    --exclude='results/' \
    --exclude='extracted/' \
    "$LOCAL/" "$REMOTE:$REMOTE_DIR/"

# ── 3. Sync small extracted files (fast) ────────────────────────────────────
echo "[3/4] Syncing extracted metadata (pkl, json)..."
rsync -avz --progress \
    "$LOCAL/extracted/"*.pkl \
    "$LOCAL/extracted/"*.json \
    "$REMOTE:$REMOTE_DIR/extracted/" 2>/dev/null || echo "  (no pkl/json files to sync)"

# ── 4. Sync data CSVs (slow — ~14 GB total) ─────────────────────────────────
echo "[4/4] Syncing data splits (~14 GB — this will take a while)..."
echo "      Transfers: X_train/val/test for binary + cat8, all y_ files"
echo "      Skipping:  ciciot_clean.csv, MERGED_CSV/, legacy_old/"
echo ""

rsync -avz --progress \
    "$LOCAL/extracted/X_train_binary.csv" \
    "$LOCAL/extracted/X_val_binary.csv" \
    "$LOCAL/extracted/X_test_binary.csv" \
    "$LOCAL/extracted/y_train_binary.csv" \
    "$LOCAL/extracted/y_val_binary.csv" \
    "$LOCAL/extracted/y_test_binary.csv" \
    "$LOCAL/extracted/X_train_cat8.csv" \
    "$LOCAL/extracted/X_val_cat8.csv" \
    "$LOCAL/extracted/X_test_cat8.csv" \
    "$LOCAL/extracted/y_train_cat8.csv" \
    "$LOCAL/extracted/y_val_cat8.csv" \
    "$LOCAL/extracted/y_test_cat8.csv" \
    "$REMOTE:$REMOTE_DIR/extracted/"

echo ""
echo "=== Sync complete ==="
echo ""
echo "Next steps on the cluster:"
echo "  1. ssh $REMOTE"
echo "  2. cd ~/CICIoT2023"
echo "  3. bash slurm/env_setup.sh           # one-time setup on login node"
echo "  4. BALANCE_JOB=\$(sbatch --parsable slurm/run_03_balance.sh)"
echo "     sbatch --dependency=afterok:\$BALANCE_JOB slurm/run_06_multiclass_ml.sh"
echo "     sbatch --dependency=afterok:\$BALANCE_JOB slurm/run_07_multiclass_dl.sh"
echo "  5. squeue                             # monitor jobs"
