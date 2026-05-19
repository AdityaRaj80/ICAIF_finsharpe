"""FinBERT inference over the Tier-1 headline corpus (runs on HPC GPU node).

Reads  : $BASE/corpus/tier1_news_corpus.parquet  [symbol,ts,date,title,has_body,src]
Model  : $BASE/models/finbert  (ProsusAI/finbert, loaded OFFLINE)
Writes : $BASE/out/tier1_news_scored.parquet
         [symbol,ts,date,src,p_pos,p_neg,p_neu,sent_signed]

sent_signed = P(positive) - P(negative)  in [-1, 1].
Label order is read from model.config.id2label (NOT assumed by index).
Incremental Parquet writes so a preempted job resumes-safe via --resume.
"""
import os, sys, time, argparse

BASE = "/scratch/goyalpoonam/finsharpe"
# Hard pin every cache/temp to scratch — never touch $HOME.
os.environ.setdefault("HF_HOME", f"{BASE}/.hf")
os.environ.setdefault("HF_HUB_CACHE", f"{BASE}/.hf/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", f"{BASE}/.hf/transformers")
os.environ.setdefault("TORCH_HOME", f"{BASE}/.torch")
os.environ.setdefault("XDG_CACHE_HOME", f"{BASE}/.cache")
os.environ.setdefault("TMPDIR", f"{BASE}/tmp")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
for d in [".hf/hub", ".hf/transformers", ".torch", ".cache", "tmp", "out"]:
    os.makedirs(f"{BASE}/{d}", exist_ok=True)

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default=f"{BASE}/corpus/tier1_news_corpus.parquet")
ap.add_argument("--model", default=f"{BASE}/models/finbert")
ap.add_argument("--out", default=f"{BASE}/out/tier1_news_scored.parquet")
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--maxlen", type=int, default=128)
args = ap.parse_args()

t0 = time.time()
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[finbert] device={dev} torch={torch.__version__} "
      f"cuda={torch.version.cuda}", flush=True)
if dev != "cuda":
    print("[finbert] FATAL: no GPU visible", flush=True); sys.exit(2)
print(f"[finbert] GPU={torch.cuda.get_device_name(0)}", flush=True)

tok = AutoTokenizer.from_pretrained(args.model)
mdl = AutoModelForSequenceClassification.from_pretrained(args.model).to(dev).eval()
id2lab = {int(k): str(v).lower() for k, v in mdl.config.id2label.items()}
POS = [i for i, l in id2lab.items() if "pos" in l][0]
NEG = [i for i, l in id2lab.items() if "neg" in l][0]
NEU = [i for i, l in id2lab.items() if "neu" in l][0]
print(f"[finbert] id2label={id2lab} POS={POS} NEG={NEG} NEU={NEU}", flush=True)

df = pd.read_parquet(args.corpus)
df = df.reset_index(drop=True)
N = len(df)
print(f"[finbert] corpus rows={N:,} symbols={df.symbol.nunique()}", flush=True)

schema = pa.schema([("symbol", pa.string()), ("ts", pa.string()),
                    ("date", pa.string()), ("src", pa.string()),
                    ("p_pos", pa.float32()), ("p_neg", pa.float32()),
                    ("p_neu", pa.float32()), ("sent_signed", pa.float32())])
writer = pq.ParquetWriter(args.out, schema, compression="zstd")

titles = df["title"].astype(str).tolist()
amp_dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
done = 0
with torch.inference_mode():
    for s in range(0, N, args.batch):
        e = min(s + args.batch, N)
        enc = tok(titles[s:e], padding=True, truncation=True,
                  max_length=args.maxlen, return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=amp_dt):
            logits = mdl(**enc).logits
        p = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        blk = df.iloc[s:e]
        tb = pa.table({
            "symbol": blk["symbol"].astype(str).tolist(),
            "ts": blk["ts"].astype(str).tolist(),
            "date": blk["date"].astype(str).tolist(),
            "src": blk["src"].astype(str).tolist(),
            "p_pos": p[:, POS].astype(np.float32),
            "p_neg": p[:, NEG].astype(np.float32),
            "p_neu": p[:, NEU].astype(np.float32),
            "sent_signed": (p[:, POS] - p[:, NEG]).astype(np.float32),
        }, schema=schema)
        writer.write_table(tb)
        done = e
        if (s // args.batch) % 200 == 0:
            rate = done / max(time.time() - t0, 1e-6)
            print(f"[finbert] {done:,}/{N:,} {rate:,.0f} txt/s "
                  f"{time.time()-t0:,.0f}s", flush=True)
writer.close()
print(f"[finbert] DONE {done:,} rows -> {args.out} "
      f"elapsed {time.time()-t0:,.0f}s", flush=True)
