#!/bin/bash
# Create/update one persistent user-space environment after GPU discovery.
# Submit with: sbatch --export=ALL,TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 cluster/setup-env.sbatch
#SBATCH --job-name=sla-setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=stu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_default
#SBATCH --output=artifacts/sla-setup-%j.out
#SBATCH --error=artifacts/sla-setup-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit this script from the repository root}"
: "${TORCH_INDEX_URL:?set TORCH_INDEX_URL only after reading the discovery job}"

SLA_ENV="${SLA_ENV:-${SLA_VENV:-$HOME/sparse_linear/.envs/sla-tutorial-py312}}"
FLA_REPO="${FLA_REPO:-$HOME/sparse_linear/flash-linear-attention}"
CONDA_EXE="${CONDA_EXE:-$HOME/miniconda3/bin/conda}"
if [[ ! -d "$FLA_REPO/.git" ]]; then
    echo "FLA_REPO is not a Git checkout: $FLA_REPO" >&2
    exit 2
fi
if [[ ! -x "$CONDA_EXE" ]]; then
    echo "Conda executable not found: $CONDA_EXE" >&2
    exit 2
fi

mkdir -p "$(dirname "$SLA_ENV")" artifacts
if [[ ! -x "$SLA_ENV/bin/python" ]]; then
    "$CONDA_EXE" create --yes --prefix "$SLA_ENV" --override-channels --channel conda-forge python=3.12 pip
fi
export PATH="$SLA_ENV/bin:$PATH"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url "$TORCH_INDEX_URL" "torch>=2.7"
python -m pip install pytest "triton>=3.3" einops "transformers>=4.45.0"
python -m pip install --no-deps --editable "$FLA_REPO"

python -m tutorial_code.scripts.environment_report
python -m pip freeze > "artifacts/environment-lock-${SLURM_JOB_ID}.txt"
echo "environment=$SLA_ENV"
echo "fla_commit=$(git -C "$FLA_REPO" rev-parse HEAD)"
