#!/bin/bash
# Run on the BITS HPC login node (has internet). Prepares the scratch
# workspace and pre-downloads FinBERT so the GPU job can run offline.
set -e
BASE=/scratch/goyalpoonam/finsharpe
mkdir -p "$BASE"/models "$BASE"/corpus "$BASE"/jobs "$BASE"/logs "$BASE"/out
echo "WORKSPACE $BASE"
ls -ld "$BASE"

PY=/home/goyalpoonam/.conda/envs/sr_opt/bin/python
"$PY" - <<'PYEOF'
import torch, transformers, sys
print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda_build", torch.version.cuda)
print("transformers", transformers.__version__)
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="ProsusAI/finbert",
                      local_dir="/scratch/goyalpoonam/finsharpe/models/finbert")
import os
print("MODEL_AT", p)
print("files", sorted(os.listdir(p)))
# sanity: load offline
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained(p)
mdl = AutoModelForSequenceClassification.from_pretrained(p)
print("LABELS", mdl.config.id2label, "n_params_M", round(mdl.num_parameters()/1e6,1))
PYEOF

du -sh /scratch/goyalpoonam/finsharpe/models/finbert
echo "PREP_OK"
