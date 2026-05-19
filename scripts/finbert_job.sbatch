#!/bin/bash
#SBATCH --job-name=fb_finsharpe
#SBATCH --account=bits
#SBATCH --partition=gpu_a100_8
#SBATCH --qos=qos_gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/goyalpoonam/finsharpe/logs/finbert_%j.out
#SBATCH --error=/scratch/goyalpoonam/finsharpe/logs/finbert_%j.err

# ---- everything pinned to /scratch; nothing in $HOME ----
BASE=/scratch/goyalpoonam/finsharpe
export HF_HOME=$BASE/.hf
export HF_HUB_CACHE=$BASE/.hf/hub
export TRANSFORMERS_CACHE=$BASE/.hf/transformers
export TORCH_HOME=$BASE/.torch
export XDG_CACHE_HOME=$BASE/.cache
export TMPDIR=$BASE/tmp
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONPATH=$BASE/pylibs:$PYTHONPATH   # pyarrow lives here (scratch, not HOME/shared env)
mkdir -p "$BASE"/.hf/hub "$BASE"/.hf/transformers "$BASE"/.torch "$BASE"/.cache "$BASE"/tmp "$BASE"/out

echo "host=$(hostname) date=$(date) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
PY=/home/goyalpoonam/.conda/envs/sr_opt/bin/python
echo "HOME usage before: $(lfs quota -hq -u goyalpoonam /home 2>/dev/null | awk '{print $2}')"

srun "$PY" "$BASE/jobs/infer_finbert.py" \
  --corpus "$BASE/corpus/tier1_news_corpus.parquet" \
  --model  "$BASE/models/finbert" \
  --out    "$BASE/out/tier1_news_scored.parquet" \
  --batch 256 --maxlen 128

rc=$?
echo "HOME usage after:  $(lfs quota -hq -u goyalpoonam /home 2>/dev/null | awk '{print $2}')"
echo "exit_code=$rc"
exit $rc
