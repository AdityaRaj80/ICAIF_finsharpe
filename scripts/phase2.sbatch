#!/bin/bash
#SBATCH --job-name=p2
#SBATCH --account=bits
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/goyalpoonam/finsharpe/icaif2026/logs/%x_%A_%a.out
#SBATCH --error=/scratch/goyalpoonam/finsharpe/icaif2026/logs/%x_%A_%a.err
# Generic Phase-2 array worker. Submitted by phase2_submit.sh with:
#   PHASE={hpo|eval}  (env, exported via --export)
# partition / qos / array range / %throttle / time are set on the
# sbatch command line so the SAME script serves H100, A100 and V100.
set -euo pipefail
BASE=/scratch/goyalpoonam/finsharpe
ROOT=$BASE/icaif2026
export P2_PANEL=$ROOT/panel/features.parquet
export P2_SYMS=$ROOT/universe/tier1.txt
export P2_OUT=$ROOT/p2out
export P2_H=${P2_H:-5}
export PYTHONPATH=$BASE/pylibs:$ROOT/engine
export PYTHONNOUSERSITE=1
# everything cache-side on scratch; nothing in $HOME
export HF_HOME=$BASE/.hf TORCH_HOME=$BASE/.torch
export XDG_CACHE_HOME=$BASE/.cache TMPDIR=$BASE/tmp
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$ROOT/logs" "$ROOT/p2out/scores" "$ROOT/p2out/selected" \
         "$BASE/tmp" "$BASE/.cache"
PY=/home/goyalpoonam/.conda/envs/sr_opt/bin/python
cd "$ROOT"
echo "host=$(hostname) job=$SLURM_JOB_ID arr=$SLURM_ARRAY_TASK_ID "\
"phase=$PHASE H=$P2_H part=$SLURM_JOB_PARTITION $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
IDX=$((SLURM_ARRAY_TASK_ID + ${IDX_OFFSET:-0}))
echo "dispatch idx=$IDX  (array_id=$SLURM_ARRAY_TASK_ID + off="\
"${IDX_OFFSET:-0})"
srun "$PY" engine/phase2.py dispatch --phase "$PHASE" --idx "$IDX"
echo "exit=$? $(date -u +%FT%TZ)"
