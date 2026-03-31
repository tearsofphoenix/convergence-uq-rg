#!/bin/bash
# Research environment setup script
# Run once on each machine

set -e

echo "===== Research Environment Setup ====="

# Check we're on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "This script is designed for macOS"
    exit 1
fi

# Detect machine type
MACHINE=$(hostname)
echo "Machine: $MACHINE"

# Base conda environment
CONDA_ENV="hybridqml311"
PYTHON_VERSION="3.11"

echo ""
echo "[1/4] Checking conda..."
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Install from https://docs.conda.io/"
    exit 1
fi

echo ""
echo "[2/4] Creating conda environment..."
conda create -n $CONDA_ENV python=$PYTHON_VERSION -y
conda activate $CONDA_ENV

echo ""
echo "[3/4] Installing packages..."
# Core scientific
conda install -y numpy scipy matplotlib scikit-learn pandas tqdm pyyaml

# ML
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# MLX (Apple Silicon)
pip install mlx mlx-metal

# JAX
pip install jax jaxlib

# FEM
conda install -y -c conda-forge fenics petsc4py scipy-openblas

echo ""
echo "[4/4] Verification..."
python3 -c "
import numpy; print(f'numpy: {numpy.__version__}')
import torch; print(f'torch: {torch.__version__}')
try:
    import mlx; print('mlx: installed')
except:
    print('mlx: not available (Intel machine?)')
try:
    import fenics; print('fenics: installed')
except:
    print('fenics: not available')
"

echo ""
echo "===== Setup complete! ====="
echo "Activate with: conda activate $CONDA_ENV"
echo "Set PYTHONPATH: export PYTHONPATH=/Users/isaac/clawd/research:\$PYTHONPATH"
