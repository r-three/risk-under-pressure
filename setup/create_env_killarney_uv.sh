#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --account=aip-craffel

set -e

echo "============================================"
echo "Creating pRisk-Pressure environment with UV at $(date)"
echo "============================================"

# Load modules BEFORE creating/activating venv
module load cuda/12.6
module load gcc arrow/19.0.1 python/3.11

mkdir -p logs

# Install UV if not available
if ! command -v uv &> /dev/null; then
    echo "Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "UV version: $(uv --version)"

# Install project with dependencies using uv sync
echo "Installing prisk-pressure with dependencies..."
uv sync

source .venv/bin/activate

echo "Python: $(which python)"

echo "============================================"
echo "Environment created successfully!"
echo "Activate with: source .venv/bin/activate"
echo "============================================"
