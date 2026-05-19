"""Title-vs-body FinBERT ablation (JURY Jury5#3).

Quantifies the title-only-scoring confound: on body-bearing articles, score
the TITLE vs the BODY-LEAD with the same FinBERT, compare sign agreement,
score correlation, and TRAIN-period forward-return IC of each.
Bounded: streams only All_external.csv, collects first N body-bearing
Tier-1 rows. Local GPU. Output: panel/title_body_ablation.md
"""
import os, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

os.environ.setdefault("HF_HOME", r"D:\Study\FinSharpe\.hf")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SRC = r"D:\Study\FinSharpe\DATA\Stock_news\nasdaq_exteral_data.csv"  # body-rich source
MODEL = r"D:\Study\FinSharpe\models\finbert"
TIER1 = r"D:\Study\FinSharpe\universe\tier1.txt"
PXDIR = r"D:\Study\FinSharpe\DATA\full_history"
OUT = r"D:\Study\FinSharpe\panel\title_body_ablation.md"
N = 6000
TRN0, TRN1 = pd.Timestamp("2013-01-01"), pd.Timestamp("2019-12-31")
t0 = time.time()

tier1 = set(pd.read_csv(TIER1, header=None)[0].astype(str).str.strip())
rows = []
for ch in pd.read_csv(SRC, usecols=["Date", "Stock_symbol", "Article_title",
                       "Article"], dtype=str, chunksize=200_000, engine="c",
                       on_bad_lines="skip", encoding_errors="replace"):
    ch = ch.dropna(subset=["Stock_symbol", "Date", "Article_title", "Article"])
    ch = ch[ch["Stock_symbol"].str.strip().isin(tier1)]
    ch = ch[ch["Article"].str.len() > 200]
    for s, d, t, a in zip(ch["Stock_symbol"].str.strip(), ch["Date"],
                          ch["Article_title"].str.strip(),
                          ch["Article"].str.strip()):
        rows.append((s, d[:10], t, a[:2000]))
        if len(rows) >= N:
            break
    if len(rows) >= N:
        break
df = pd.DataFrame(rows, columns=["symbol", "date", "title", "body"])
print(f"[ablate] collected {len(df)} body-bearing rows "
      f"{time.time()-t0:,.0f}s", flush=True)

dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
mdl = AutoModelForSequenceClassification.from_pretrained(MODEL).to(dev).eval()
i2l = {int(k): str(v).lower() for k, v in mdl.config.id2label.items()}
POS = [i for i, l in i2l.items() if "pos" in l][0]
NEG = [i for i, l in i2l.items() if "neg" in l][0]


@torch.inference_mode()
def score(texts):
    out = []
    for s in range(0, len(texts), 128):
        e = tok(texts[s:s + 128], padding=True, truncation=True,
                max_length=256, return_tensors="pt").to(dev)
        p = torch.softmax(mdl(**e).logits.float(), -1).cpu().numpy()
        out.append(p[:, POS] - p[:, NEG])
    return np.concatenate(out)


df["s_title"] = score(df["title"].tolist())
df["s_body"] = score(df["body"].tolist())
sign_agree = float((np.sign(df.s_title) == np.sign(df.s_body)).mean())
rho = float(spearmanr(df.s_title, df.s_body)[0])

# train-period forward-5d return IC for each text unit
df["d"] = pd.to_datetime(df["date"], errors="coerce")
tr = df[(df.d >= TRN0) & (df.d <= TRN1)].copy()
ics = {}
for sym in tr.symbol.unique():
    f = os.path.join(PXDIR, f"{sym}.csv")
    if not os.path.exists(f):
        continue
    px = pd.read_csv(f, usecols=["date", "adj close"])
    px["date"] = pd.to_datetime(px["date"], errors="coerce")
    px = px.dropna().sort_values("date").drop_duplicates("date")
    c = px["adj close"].astype("float64")
    px["fwd5"] = np.log(c.shift(-5) / c)
    g = tr[tr.symbol == sym].merge(px, left_on="d", right_on="date",
                                   how="inner").dropna(subset=["fwd5"])
    if len(g) >= 10:
        ics.setdefault("t", []).append(g[["s_title", "fwd5"]])
        ics.setdefault("b", []).append(g[["s_body", "fwd5"]])
T = pd.concat(ics["t"]) if ics.get("t") else pd.DataFrame()
B = pd.concat(ics["b"]) if ics.get("b") else pd.DataFrame()
ic_t = spearmanr(T.s_title, T.fwd5)[0] if len(T) else np.nan
ic_b = spearmanr(B.s_body, B.fwd5)[0] if len(B) else np.nan

verdict_txt = ("Title and body largely agree; title-only scoring is a "
               "defensible simplification (and most FNSPID bodies are "
               "absent anyway)." if sign_agree > 0.7 else
               "Title/body diverge, but C3 is DROPPED — sentiment is only "
               "a de-scoped optional feature; this is advisory and rides "
               "on no claim.")
R = ["# Title-vs-Body FinBERT Ablation (Jury5#3)\n",
     f"_n={len(df)} body-bearing All_external Tier-1 articles; local GPU._\n",
     f"- title/body sign agreement: **{sign_agree:.3f}**",
     f"- title/body score Spearman: **{rho:.3f}**",
     f"- TRAIN fwd-5d return IC: title **{ic_t:+.4f}** vs body "
     f"**{ic_b:+.4f}** (n_title={len(T):,}, n_body={len(B):,})",
     "\n## Verdict\n",
     f"- {verdict_txt}",
     "- Advisory only (C3 dropped, PREREG §10): sentiment is a de-scoped "
     "optional feature; title chosen a-priori (51% bodies absent).",
     f"\n_elapsed {time.time()-t0:,.0f}s_"]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w", encoding="utf-8").write("\n".join(R))
print(f"[ablate] sign_agree={sign_agree:.3f} rho={rho:.3f} "
      f"ic_title={ic_t:+.4f} ic_body={ic_b:+.4f} ({time.time()-t0:,.0f}s)")
print("wrote", OUT)
