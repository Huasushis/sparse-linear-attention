#!/bin/bash
# Discover profiling tools and create one small, bounded attention profile.
# Optional: sbatch --export=ALL,RUN_TORCH_PROFILER=0,RUN_NSYS=1 cluster/profile-attention.sbatch
# Optional: sbatch --export=ALL,RUN_TORCH_PROFILER=0,RUN_NCU=1 cluster/profile-attention.sbatch
#SBATCH --job-name=sla-profile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --account=stu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_default
#SBATCH --output=artifacts/sla-profile-%j.out
#SBATCH --error=artifacts/sla-profile-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit this script from the repository root}"
cd .
REPO_ROOT="$(pwd -P)"
mkdir -p artifacts

SLA_ENV="${SLA_ENV:-${SLA_VENV:-$HOME/sparse_linear/.envs/sla-tutorial-py312}}"
if [[ ! -x "$SLA_ENV/bin/python" ]]; then
    echo "Missing environment: $SLA_ENV; run cluster/setup-env.sbatch first" >&2
    exit 2
fi
export PATH="$SLA_ENV/bin:$PATH"
JOB_TMP="${SLURM_TMPDIR:-/tmp/sla-profile-${USER}-${SLURM_JOB_ID}}"
mkdir -p "$JOB_TMP"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
echo "repo_root=$REPO_ROOT"
echo "job_tmp=$JOB_TMP"
cd "$JOB_TMP"

echo "=== Tool discovery ==="
for tool in nvidia-smi nsys ncu compute-sanitizer nvprof; do
    if command -v "$tool" >/dev/null 2>&1; then
        printf '%-20s %s\n' "$tool" "$(command -v "$tool")"
    else
        printf '%-20s %s\n' "$tool" "NOT FOUND"
    fi
done
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python -m tutorial_code.scripts.environment_report

PROFILE_ARGS=(
    --operator "${PROFILE_OPERATOR:-torch_sdpa}"
    --mode "${PROFILE_MODE:-prefill}"
    --seq-len "${PROFILE_SEQ_LEN:-512}"
    --dtype "${PROFILE_DTYPE:-bfloat16}"
    --warmup "${PROFILE_WARMUP:-10}"
)

if [[ "${RUN_TORCH_PROFILER:-1}" == "1" ]]; then
    echo "=== PyTorch profiler ==="
    python -m tutorial_code.profiling.profile_attention \
        "${PROFILE_ARGS[@]}" \
        --steps 5 \
        --trace "$JOB_TMP/torch-profiler-${SLURM_JOB_ID}.json" \
        | tee "$REPO_ROOT/artifacts/torch-profiler-summary-${SLURM_JOB_ID}.txt"
    cp "$JOB_TMP/torch-profiler-${SLURM_JOB_ID}.json" "$REPO_ROOT/artifacts/"
fi

if [[ "${RUN_NSYS:-0}" == "1" ]]; then
    if ! command -v nsys >/dev/null 2>&1; then
        echo "RUN_NSYS=1 but nsys is not available on this node" >&2
        exit 3
    fi
    echo "=== Nsight Systems ==="
    nsys profile \
        --trace=cuda,nvtx,osrt \
        --sample=none \
        --cpuctxsw=none \
        -o "$JOB_TMP/nsys-attention-${SLURM_JOB_ID}" \
        python -m tutorial_code.profiling.profile_attention "${PROFILE_ARGS[@]}" --steps 5
    cp "$JOB_TMP/nsys-attention-${SLURM_JOB_ID}.nsys-rep" "$REPO_ROOT/artifacts/"
fi

if [[ "${RUN_NCU:-0}" == "1" ]]; then
    if ! command -v ncu >/dev/null 2>&1; then
        echo "RUN_NCU=1 but ncu is not available on this node" >&2
        exit 3
    fi
    echo "=== Nsight Compute ==="
    ncu --version
    NCU_CHIPS="$(ncu --list-chips)"
    echo "supported_chips=$NCU_CHIPS"
    if [[ "$GPU_NAME" == *"RTX 5090"* && "$NCU_CHIPS" != *"gb202"* ]]; then
        echo "ncu_status=SKIPPED: installed ncu does not list the RTX 5090 gb202 chip"
        exit 0
    fi
    ncu \
        --set "${NCU_SET:-default}" \
        --target-processes all \
        --nvtx \
        --nvtx-include "sla_attention_step/" \
        --launch-count 1 \
        --page details \
        -o "$JOB_TMP/ncu-attention-${SLURM_JOB_ID}" \
        python -m tutorial_code.profiling.profile_attention "${PROFILE_ARGS[@]}" --steps 1
    cp "$JOB_TMP/ncu-attention-${SLURM_JOB_ID}.ncu-rep" "$REPO_ROOT/artifacts/"
fi
