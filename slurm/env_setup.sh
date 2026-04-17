#!/bin/bash
# =============================================================================
# env_setup.sh — One-time conda environment setup
#
# Run this MANUALLY on the LOGIN NODE (not as a SLURM job).
# Only needs to be run once per cluster account.
#
# Usage:
#   ssh user@login-node
#   cd ~/CICIoT2023
#   bash slurm/env_setup.sh
# =============================================================================

set -euo pipefail

ENV_PATH="$HOME/ciciot"

echo "=== Loading miniconda module ==="
module load miniconda/miniconda3

# Create env in home folder (required by cluster policy)
if [ -d "$ENV_PATH" ]; then
    echo "Environment already exists at $ENV_PATH — skipping creation."
else
    echo "=== Creating conda environment at $ENV_PATH (Python 3.10) ==="
    conda create -p "$ENV_PATH" python=3.10 -y
fi

echo "=== Installing dependencies ==="
PYTHON="$ENV_PATH/bin/python"
PIP="$ENV_PATH/bin/pip"

$PIP install --upgrade pip

# Core data science
$PIP install "numpy>=1.24" "pandas>=2.0" "scikit-learn>=1.3"

# ML models
$PIP install "xgboost>=2.0"

# Class imbalance
$PIP install "imbalanced-learn>=0.11"

# PyTorch — CUDA 12.8 (RTX 5070 Ti requires 12.8+, default pip install covers this)
$PIP install torch torchvision

# TensorFlow with bundled CUDA (avoids system CUDA version conflicts)
$PIP install "tensorflow[and-cuda]"

# Visualization
$PIP install "matplotlib>=3.7" "seaborn>=0.12"

# Install the ciciot_ids package in editable mode
$PIP install -e "$HOME/CICIoT2023"

echo ""
echo "=== Setup complete ==="
echo "Python path: $PYTHON"
$PYTHON --version
echo ""
echo "Test with:"
echo "  $PYTHON -c \"import torch; import tensorflow as tf; print('torch:', torch.__version__, 'tf:', tf.__version__)\""
