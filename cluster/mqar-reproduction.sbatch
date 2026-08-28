#!/bin/bash
# Two-layer GDN/KDA MQAR learning-curve reproduction.
#SBATCH --job-name=sla-mqar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --account=stu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_default
#SBATCH --output=artifacts/sla-mqar-%j.out
#SBATCH --error=artifacts/sla-mqar-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit this script from the repository root}"
cd .
REPO_ROOT="$(pwd -P)"
SLA_ENV="${SLA_ENV:-${SLA_VENV:-$HOME/sparse_linear/.envs/sla-tutorial-py312}}"
FLA_REPO="${FLA_REPO:-$HOME/sparse_linear/flash-linear-attention}"
export PATH="$SLA_ENV/bin:$PATH"
export FLA_REPO
export PYTHONPATH="$REPO_ROOT:$FLA_REPO${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_F32_DEFAULT=ieee
RESULT_DIR="$REPO_ROOT/artifacts/mqar-${SLURM_JOB_ID}"
mkdir -p "$RESULT_DIR"

for attempt in 1 2; do
    if python -m tutorial_code.experiments.mqar_gdn_kda \
        --architectures gdn kda \
        --steps 8000 --batch-size 16 \
        --learning-rate 5e-4 --log-every 100 --eval-every 500 \
        --output "$RESULT_DIR/mqar.json"; then
        echo "result_dir=$RESULT_DIR"
        exit 0
    fi
    echo "attempt $attempt failed; refreshing NFS metadata" >&2
    cd .
    ls -l "$SLA_ENV/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so" >/dev/null
done
exit 1
