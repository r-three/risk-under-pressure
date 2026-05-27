#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --partition=debug
#SBATCH --account=def-craffel

set -e

echo "============================================"
echo "Creating pRisk-Pressure environment with UV at $(date)"
echo "============================================"

module load StdEnv/2023
module load cuda/12.6
module load gcc arrow/19.0.1 python/3.11

mkdir -p logs

if ! command -v uv &> /dev/null; then
    echo "Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "UV version: $(uv --version)"

echo "Installing risk-under-pressure with dependencies..."
uv sync

source .venv/bin/activate

echo "Python: $(which python)"

echo "============================================"
echo "Environment created successfully!"
echo "Activate with: source .venv/bin/activate"
echo "============================================"
