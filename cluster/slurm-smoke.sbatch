#!/bin/bash
# A short tutorial job: environment evidence + correctness tests + tiny benchmark.
# Submit from the remote repository root with: sbatch cluster/slurm-smoke.sbatch
# Copy account/partition/QoS from your site's template if they differ.
#SBATCH --job-name=sla-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --account=stu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_default
#SBATCH --output=artifacts/sla-smoke-%j.out
#SBATCH --error=artifacts/sla-smoke-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit this script from the repository root}"
mkdir -p artifacts
SLA_VENV="${SLA_VENV:-$HOME/sparse_linear/.venvs/sla-tutorial-py312}"
if [[ ! -x "$SLA_VENV/bin/python" ]]; then
    echo "Missing environment: $SLA_VENV; run cluster/setup-env.sbatch first" >&2
    exit 2
fi
source "$SLA_VENV/bin/activate"

echo "=== Git commit ==="
git rev-parse HEAD || true
echo "=== Environment ==="
python -m tutorial_code.scripts.environment_report
echo "=== Correctness tests ==="
python -m pytest tutorial_code/tests -q
echo "=== GDN/KDA scalar-gate probe ==="
python -m tutorial_code.scripts.gdn_kda_probe
echo "=== Small attention benchmark ==="
python -m tutorial_code.benchmarks.benchmark_attention --operator dense --mode prefill --device cuda --dtype bfloat16 --seq-len 512 --repeats 20 --output "artifacts/dense-prefill-${SLURM_JOB_ID}.json"
python -m tutorial_code.benchmarks.benchmark_attention --operator dense --mode decode --device cuda --dtype bfloat16 --seq-len 512 --repeats 20 --output "artifacts/dense-decode-${SLURM_JOB_ID}.json"
