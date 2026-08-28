#!/bin/bash
# NSA long-sequence rerun at the dimensions supported by the pinned FLA kernel.
#SBATCH --job-name=sla-nsa-repro
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --account=stu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_default
#SBATCH --output=artifacts/sla-nsa-repro-%j.out
#SBATCH --error=artifacts/sla-nsa-repro-%j.err

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
RESULT_DIR="$REPO_ROOT/artifacts/nsa-${SLURM_JOB_ID}"
mkdir -p "$RESULT_DIR"

for attempt in 1 2; do
    if python -m tutorial_code.benchmarks.paper_operator_reproduction \
        --study nsa --modes fwd fwdbwd \
        --nsa-key-dim 128 --nsa-value-dim 128 \
        --warmup-ms 25 --rep-ms 100 \
        --output "$RESULT_DIR/nsa.json"; then
        echo "result_dir=$RESULT_DIR"
        exit 0
    fi
    echo "attempt $attempt failed; refreshing NFS metadata" >&2
    cd .
done
exit 1
