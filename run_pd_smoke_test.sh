#!/usr/bin/env bash
#SBATCH --job-name=smoke_test_pd_http
#SBATCH --qos=blanca-clearlab1
#SBATCH --account=blanca-clearlab1
#SBATCH --nodelist=bgpu-g4-u30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=outputs/%j.out
#SBATCH --error=outputs/%j.err


set -euo pipefail

mkdir -p outputs

echo "job id: ${SLURM_JOB_ID:-unknown}"
echo "host: $(hostname)"
echo "start: $(date)"
echo "pwd: $(pwd)"

source /scratch/alpine/jana7431/Intune/jana7431-gpt/bin/activate

python3 - <<'PY'
import sys
import torch
print("python:", sys.executable)
print("torch:", torch.__version__)
PY

echo "Running PD HTTP smoke test..."
python3 -m unittest -v tests.test_pd_http_smoke
echo "end: $(date)"
