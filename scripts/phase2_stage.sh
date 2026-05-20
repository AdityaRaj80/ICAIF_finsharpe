#!/bin/bash
# RUN ON THE BITS HPC LOGIN NODE (has internet). Stages the Phase-2
# workspace under /scratch ONLY (never $HOME) and installs optuna into
# the scratch pylibs dir (pip --target; no HOME writes). Expects the
# source tarball phase2_src.tgz and panel/features.parquet to have been
# copied to $BASE/icaif2026/ already (see phase2_push.sh, run locally).
set -euo pipefail
BASE=/scratch/goyalpoonam/finsharpe
ROOT=$BASE/icaif2026
PYLIBS=$BASE/pylibs
PY=/home/goyalpoonam/.conda/envs/sr_opt/bin/python

mkdir -p "$ROOT"/{engine,panel,universe,scripts,logs,p2out/scores,p2out/selected}
cd "$ROOT"

if [ -f phase2_src.tgz ]; then
  tar xzf phase2_src.tgz                       # -> engine/ universe/ scripts/
  echo "STAGED src: $(ls engine | wc -l) engine files"
fi
test -f panel/features.parquet || { echo "MISSING panel/features.parquet"; exit 2; }
test -f PREREGISTRATION.md && test -f PREREGISTRATION.sha256 \
  || { echo "MISSING PREREGISTRATION.{md,sha256}"; exit 2; }

# optuna (+deps) into scratch pylibs; NEVER $HOME
export PYTHONNOUSERSITE=1
"$PY" -m pip install --no-input --target="$PYLIBS" \
  "optuna>=3.6" 2>&1 | tail -3

PYTHONPATH="$PYLIBS" "$PY" - <<'PYEOF'
import optuna, pyarrow, torch, sys
print("optuna", optuna.__version__, "pyarrow", pyarrow.__version__,
      "torch", torch.__version__, "py", sys.version.split()[0])
PYEOF

# integrity gate BEFORE any training: live doc must match recorded SHA
cd "$ROOT"
PYTHONPATH="$PYLIBS" P2_PANEL="$ROOT/panel/features.parquet" \
  "$PY" engine/check_prereg_constants.py | tail -1
echo "STAGE_OK  root=$ROOT"
