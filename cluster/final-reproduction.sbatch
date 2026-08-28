#!/bin/bash
# Final operator-level reproduction matrix for the report.
# Submit from the tutorial repository root.
#SBATCH --job-name=sla-final-repro
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --account=stu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_default
#SBATCH --output=artifacts/sla-final-repro-%j.out
#SBATCH --error=artifacts/sla-final-repro-%j.err

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
RESULT_DIR="$REPO_ROOT/artifacts/final-${SLURM_JOB_ID}"
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
run_python -m tutorial_code.scripts.environment_report
run_python -m pytest tutorial_code/tests tutorial_code/graders/test_exercise_dense.py tutorial_code/graders/test_exercise_triton.py -q
run_python -m tutorial_code.scripts.fla_operator_probe | tee "$RESULT_DIR/fla-operator-probe.json"

for length in 512 2048 8192; do
    for mode in prefill decode; do
        run_python -m tutorial_code.benchmarks.benchmark_attention \
            --operator dense --mode "$mode" --device cuda --dtype bfloat16 \
            --seq-len "$length" --warmup 10 --repeats 20 \
            --output "$RESULT_DIR/dense-${mode}-${length}.json"
        run_python -m tutorial_code.benchmarks.benchmark_attention \
            --operator linear --mode "$mode" --device cuda --dtype bfloat16 \
            --seq-len "$length" --warmup 10 --repeats 20 \
            --output "$RESULT_DIR/linear-${mode}-${length}.json"
    done
done

run_python -m tutorial_code.benchmarks.benchmark_fla_ops \
    --seq-lens 512 2048 8192 --warmup 5 --repeats 20 \
    --output "$RESULT_DIR/fla-gdn-kda-nsa.json"

echo "result_dir=$RESULT_DIR"
