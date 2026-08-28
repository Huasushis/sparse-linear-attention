#!/bin/bash
# Paper-scale operator correctness, latency, backward, and memory reproduction.
# Submit from the sparse-linear-attention repository root.
#SBATCH --job-name=sla-paper-repro
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --account=stu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_default
#SBATCH --output=artifacts/sla-paper-repro-%j.out
#SBATCH --error=artifacts/sla-paper-repro-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit this script from the repository root}"
cd .
REPO_ROOT="$(pwd -P)"
SLA_ENV="${SLA_ENV:-${SLA_VENV:-$HOME/sparse_linear/.envs/sla-tutorial-py312}}"
FLA_REPO="${FLA_REPO:-$HOME/sparse_linear/flash-linear-attention}"
if [[ ! -x "$SLA_ENV/bin/python" ]]; then
    echo "Missing environment: $SLA_ENV" >&2
    exit 2
fi
if [[ ! -d "$FLA_REPO/.git" ]]; then
    echo "Missing FLA checkout: $FLA_REPO" >&2
    exit 2
fi
export PATH="$SLA_ENV/bin:$PATH"
export FLA_REPO
export PYTHONPATH="$REPO_ROOT:$FLA_REPO${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_F32_DEFAULT=ieee
RESULT_DIR="$REPO_ROOT/artifacts/paper-${SLURM_JOB_ID}"
mkdir -p "$RESULT_DIR"

run_python() {
    local attempt
    for attempt in 1 2; do
        if python "$@"; then
            return 0
        fi
        echo "python attempt $attempt failed; refreshing NFS metadata" >&2
        cd .
        ls -l "$SLA_ENV/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so" >/dev/null
    done
    return 1
}

echo "tutorial_commit=$(git rev-parse HEAD)"
echo "fla_commit=$(git -C "$FLA_REPO" rev-parse HEAD)"
run_python -m tutorial_code.scripts.environment_report | tee "$RESULT_DIR/environment.json"

# These are the official FLA correctness contracts.  They compare outputs,
# final recurrent states, and input gradients against naive/reference paths.
(
    cd "$FLA_REPO"
    cd .
    run_python -m pytest -q \
        tests/ops/test_gdn.py -k 'test_chunk_with_chunk_size' \
        | tee "$RESULT_DIR/test-gdn.txt"
    run_python -m pytest -q \
        tests/ops/test_kda.py -k 'test_naive_chunk or test_chunk_invalid_chunk_size' \
        | tee "$RESULT_DIR/test-kda.txt"
    run_python -m pytest -q \
        tests/ops/test_dplr_delta.py -k 'test_chunk and not varlen' \
        | tee "$RESULT_DIR/test-dplr.txt"
    run_python -m pytest -q \
        tests/ops/test_nsa.py -k 'test_parallel and not varlen and not decode and not compressive and not topk' \
        | tee "$RESULT_DIR/test-nsa.txt"
)

for study in delta kimi nsa; do
    run_python -m tutorial_code.benchmarks.paper_operator_reproduction \
        --study "$study" --modes fwd fwdbwd --warmup-ms 25 --rep-ms 100 \
        --output "$RESULT_DIR/${study}.json"
done

echo "result_dir=$RESULT_DIR"
