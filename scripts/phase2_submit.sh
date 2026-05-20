#!/bin/bash
# RUN ON THE BITS HPC LOGIN NODE. Submits the H=5 Phase-2 campaign:
#   (1) HPO   : 17 tasks (8 backbones x {mse,risk} + ridge), a100, %3
#   (2) EVAL  : 1275 tasks (1200 deep + 75 ridge), mirrored across H100 /
#               A100 / V100 — "whichever is free" (idempotent skip makes
#               the 3 mirrors race-safe; first to a task wins, others
#               exit 0), each throttled to be a good shared-cluster
#               citizen. Depends afterok on HPO.
#   (3) AGG   : backtest.score_h1 -> bench/phase2_results.md (H1 verdict)
# Freeze (PREREG k.1) stamps automatically at the first task. Re-running
# this script is safe (idempotent skip) -> it doubles as the requeue/
# resume tool for the multi-day campaign.
set -euo pipefail
SB=/scratch/goyalpoonam/finsharpe/icaif2026/scripts/phase2.sbatch
COM="--account=bits"

hpo=$(sbatch --parsable $COM --job-name=p2hpo \
      --partition=gpu_a100_8 --qos=qos_gpu_a100 \
      --time=2-00:00:00 --array=0-16%3 \
      --export=ALL,PHASE=hpo,P2_H=5 "$SB")
echo "HPO  array submitted: $hpo  (gpu_a100_8, 17 tasks, %3)"

dep="--dependency=afterany:$hpo"   # eval defers per-cell if its HPO
                                  # config is missing; safe to re-run
# SLURM MaxArraySize=1001 -> split 0-1274 into two chunks per partition.
submit_eval () {                  # $1=part $2=qosflag $3=throttle $4=tag
  local p=$1 q=$2 thr=$3 tag=$4
  local a b
  a=$(sbatch --parsable $COM $dep --job-name=p2ev_${tag}A \
        --partition=$p $q --time=12:00:00 --array=0-999%$thr \
        --export=ALL,PHASE=eval,P2_H=5,IDX_OFFSET=0 "$SB") || a=""
  b=$(sbatch --parsable $COM $dep --job-name=p2ev_${tag}B \
        --partition=$p $q --time=12:00:00 --array=0-274%$thr \
        --export=ALL,PHASE=eval,P2_H=5,IDX_OFFSET=1000 "$SB") || b=""
  echo "$a $b"
}
read e_hA e_hB < <(submit_eval gpu_h100_4 ""                          4 h)
read e_aA e_aB < <(submit_eval gpu_a100_8 "--qos=qos_gpu_a100"        3 a)
read e_vA e_vB < <(submit_eval gpu_v100_2 ""                          2 v)
echo "EVAL arrays: h100=$e_hA,$e_hB  a100=$e_aA,$e_aB  v100=$e_vA,$e_vB"

aggdep="afterany"
for j in $e_hA $e_hB $e_aA $e_aB $e_vA $e_vB; do
  [ -n "$j" ] && aggdep="$aggdep:$j"
done
agg=$(sbatch --parsable $COM --dependency=$aggdep \
      --job-name=p2agg --partition=gpu_v100_2 --time=02:00:00 \
      --output=/scratch/goyalpoonam/finsharpe/icaif2026/logs/p2agg_%j.out \
      --wrap="cd /scratch/goyalpoonam/finsharpe/icaif2026 && \
PYTHONPATH=/scratch/goyalpoonam/finsharpe/pylibs:engine \
P2_PANEL=\$PWD/panel/features.parquet P2_OUT=\$PWD/p2out P2_H=5 \
/home/goyalpoonam/.conda/envs/sr_opt/bin/python \
engine/phase2_aggregate.py")
echo "AGG  job submitted: $agg  (afterany EVAL)"
echo "queue:"; squeue -u "$USER" -o '%.10i %.9P %.7j %.2t %.5D %R' | tail -n +1
