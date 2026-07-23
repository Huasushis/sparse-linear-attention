#!/bin/bash
# Dependency-free GPU discovery. Submit before choosing a PyTorch CUDA wheel.
#SBATCH --job-name=sla-discover
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --account=stu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_default
#SBATCH --output=artifacts/sla-discover-%j.out
#SBATCH --error=artifacts/sla-discover-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit this script from the repository root}"
mkdir -p artifacts
echo "=== host / Slurm ==="
hostname
env | grep '^SLURM_' | sort || true
echo "=== GPU ==="
nvidia-smi
echo "=== driver / CUDA query ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "=== base Python ==="
python --version
echo "=== git ==="
git rev-parse HEAD || true
