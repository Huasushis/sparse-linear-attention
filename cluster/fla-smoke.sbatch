#!/bin/bash
# Run a small FLA GDN/KDA correctness probe in the persistent tutorial env.
#SBATCH --job-name=sla-fla
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --account=stu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_default
#SBATCH --output=artifacts/sla-fla-%j.out
#SBATCH --error=artifacts/sla-fla-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit this script from the repository root}"
SLA_ENV="${SLA_ENV:-${SLA_VENV:-$HOME/sparse_linear/.envs/sla-tutorial-py312}}"
FLA_REPO="${FLA_REPO:-$HOME/sparse_linear/flash-linear-attention}"
if [[ ! -x "$SLA_ENV/bin/python" ]]; then
    echo "Missing environment: $SLA_ENV; run cluster/setup-env.sbatch first" >&2
    exit 2
fi
if [[ ! -d "$FLA_REPO/.git" ]]; then
    echo "Missing FLA checkout: $FLA_REPO" >&2
    exit 2
fi
export PATH="$SLA_ENV/bin:$PATH"
export PYTHONPATH="$SLURM_SUBMIT_DIR:$FLA_REPO${PYTHONPATH:+:$PYTHONPATH}"

echo "tutorial_commit=$(git rev-parse HEAD)"
echo "fla_commit=$(git -C "$FLA_REPO" rev-parse HEAD)"
python -m tutorial_code.scripts.environment_report
python -m tutorial_code.scripts.fla_operator_probe | tee "artifacts/fla-operator-${SLURM_JOB_ID}.json"
