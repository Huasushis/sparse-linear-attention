#!/bin/bash
# PyTorch-profiler traces for representative paper-scale operators.
#SBATCH --job-name=sla-paper-prof
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --account=stu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_default
#SBATCH --output=artifacts/sla-paper-prof-%j.out
#SBATCH --error=artifacts/sla-paper-prof-%j.err

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
RESULT_DIR="$REPO_ROOT/artifacts/profile-${SLURM_JOB_ID}"
mkdir -p "$RESULT_DIR"

for op in delta kda nsa; do
    success=0
    for attempt in 1 2; do
        if python -m tutorial_code.benchmarks.profile_paper_ops \
            --op "$op" --warmup 3 \
            --output "$RESULT_DIR/${op}.json" \
            --trace "$RESULT_DIR/${op}-trace.json"; then
            success=1
            break
        fi
        echo "$op attempt $attempt failed; refreshing NFS metadata" >&2
        cd .
    done
    [[ "$success" -eq 1 ]]
done
echo "result_dir=$RESULT_DIR"
