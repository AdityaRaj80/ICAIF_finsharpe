"""Leakage-safe causal feature panel (JURY FATAL-B / Jury2#2).

Per (symbol, trading day) in [2011-01-01, 2023-12-28]:
  * Alpha158-lite features (Reference.md §5), ALL strictly BACKWARD-looking
    (pandas trailing rolling; window ends at t, uses only data <= t).
  * OHLC made split/div-consistent via adj_close/close factor.
  * Forward targets fwd_ret_H / fwd_vol_H for H in {5,20,63,126,252}
    (labels, NaN where t+H beyond data).
  * Sentiment columns joined from the T+1-correct sentiment_daily panel.
  * Per-stock z-scoring fit on TRAIN ROWS ONLY (2013-2019), applied to all
    splits — no val/test moment ever enters normalization. Fitted (mu,sigma)
    persisted for the independent leakage QC to re-verify.
Outputs: panel/features.parquet , panel/feature_norm_stats.parquet
"""
import os, time
import numpy as np
import pandas as pd

PXDIR = r"D:\Study\FinSharpe\DATA\full_history"
SENT = r"D:\Study\FinSharpe\panel\sentiment_daily.parquet"
TIER1 = r"D:\Study\FinSharpe\universe\tier1.txt"
OUTF = r"D:\Study\FinSharpe\panel\features.parquet"
OUTN = r"D:\Study\FinSharpe\panel\feature_norm_stats.parquet"
W0, W1 = pd.Timestamp("2011-01-01"), pd.Timestamp("2023-12-28")
TRAIN = (pd.Timestamp("2013-01-01"), pd.Timestamp("2019-12-31"))
VAL = (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"))
TEST = (pd.Timestamp("2021-01-01"), pd.Timestamp("2023-12-28"))
HZ = [5, 20, 63, 126, 252]
EPS = 1e-12
t0 = time.time()

tier1 = pd.read_csv(TIER1, header=None)[0].astype(str).str.strip().tolist()
sent = pd.read_parquet(SENT)
sent_g = {s: d.set_index("date") for s, d in sent.groupby("symbol")}


def split_tag(dts):
    s = np.full(len(dts), "warmup", dtype=object)
    s[(dts >= TRAIN[0]) & (dts <= TRAIN[1])] = "train"
    s[(dts >= VAL[0]) & (dts <= VAL[1])] = "val"
    s[(dts >= TEST[0]) & (dts <= TEST[1])] = "test"
    return s


def feats_one(sym):
    fp = os.path.join(PXDIR, f"{sym}.csv")
    df = pd.read_csv(fp, usecols=["date", "open", "high", "low", "close",
                                  "adj close", "volume"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = (df.dropna(subset=["date"]).sort_values("date")
            .drop_duplicates("date"))
    df = df[(df.date >= W0) & (df.date <= W1)].reset_index(drop=True)
    if len(df) < 300:
        return None
    c0 = df["close"].clip(lower=EPS)
    f = (df["adj close"] / c0).values            # split/div factor
    o = df["open"].values * f
    h = df["high"].values * f
    l = df["low"].values * f
    c = df["adj close"].values.astype("float64")
    v = df["volume"].values.astype("float64")
    c = np.where(c <= 0, np.nan, c)
    S = pd.DataFrame({"symbol": sym, "date": df["date"].values})
    cS, oS, hS, lS = (pd.Series(c), pd.Series(o), pd.Series(h), pd.Series(l))
    vS = pd.Series(v)
    lc, lv = np.log(cS.clip(lower=EPS)), np.log(vS.clip(lower=1) + 1)

    rng = (hS - lS).replace(0, np.nan)
    S["KMID"] = (cS - oS) / oS.replace(0, np.nan)
    S["KLEN"] = (hS - lS) / oS.replace(0, np.nan)
    S["KMID2"] = (cS - oS) / (rng + EPS)
    S["KUP"] = (hS - np.maximum(oS, cS)) / oS.replace(0, np.nan)
    S["KLOW"] = (np.minimum(oS, cS) - lS) / oS.replace(0, np.nan)
    S["KSFT"] = (2 * cS - hS - lS) / oS.replace(0, np.nan)
    S["KSFT2"] = (2 * cS - hS - lS) / (rng + EPS)
    for k in [1, 5, 10, 20, 30, 60]:
        S[f"RET_{k}"] = np.log(cS / cS.shift(k))
    for w in [5, 10, 20, 30, 60]:
        S[f"MA_{w}"] = cS.rolling(w).mean() / cS
        S[f"STD_{w}"] = cS.rolling(w).std() / cS
        S[f"MAX_{w}"] = hS.rolling(w).max() / cS
        S[f"MIN_{w}"] = lS.rolling(w).min() / cS
        rmin = lS.rolling(w).min()
        S[f"RSV_{w}"] = (cS - rmin) / (hS.rolling(w).max() - rmin + EPS)
        S[f"VMA_{w}"] = vS.rolling(w).mean() / (vS + EPS)
        S[f"VSTD_{w}"] = vS.rolling(w).std() / (vS + EPS)
    for w in [10, 20, 60]:
        S[f"CORR_{w}"] = lc.rolling(w).corr(lv)
    for w in [5, 10, 20]:
        S[f"MOM_{w}"] = cS / cS.shift(w) - 1.0

    # forward targets (labels; future by definition, NOT features)
    r1 = np.log(cS / cS.shift(1))
    for H in HZ:
        S[f"fwd_ret_{H}"] = np.log(cS.shift(-H) / cS)
        S[f"fwd_vol_{H}"] = r1.shift(-1).rolling(H).std().shift(-(H - 1))

    # sentiment join (already T+1 point-in-time)
    sg = sent_g.get(sym)
    if sg is not None:
        j = sg.reindex(df["date"].values)
        sd = j["sent_decay"].values
        S["SENT"] = sd
        S["SENT_DELTA"] = pd.Series(sd) - pd.Series(sd).shift(1)
        S["SENT_MA_5"] = pd.Series(sd).rolling(5).mean()
        S["SENT_MA_20"] = pd.Series(sd).rolling(20).mean()
        S["SENT_STD_20"] = pd.Series(sd).rolling(20).std()
        S["DSN"] = j["days_since_news"].values
        S["NEWS_INT"] = np.log1p(np.nan_to_num(j["news_intensity"].values))
        S["SENT_DISP"] = np.nan_to_num(j["sent_dispersion"].values)
        S["HAS_NEWS"] = j["has_news"].fillna(False).astype(np.int8).values
    S["split"] = split_tag(df["date"].values)
    return S


parts = []
for i, sym in enumerate(tier1):
    try:
        r = feats_one(sym)
    except Exception as e:
        print(f"[feat] FAIL {sym}: {str(e)[:70]}", flush=True); continue
    if r is not None:
        parts.append(r)
    if (i + 1) % 200 == 0:
        print(f"[feat] {i+1}/{len(tier1)} {time.time()-t0:,.0f}s", flush=True)

P = pd.concat(parts, ignore_index=True)
ID = ["symbol", "date", "split"]
TGT = [c for c in P.columns if c.startswith(("fwd_ret_", "fwd_vol_"))]
RAWBIN = ["HAS_NEWS"]
FEAT = [c for c in P.columns if c not in ID + TGT + RAWBIN]

# cross-sectional rank features (per date, across universe — causal: only
# uses day-t values across stocks; no time leakage)
for src in ["RET_20", "MOM_20", "MA_20"]:
    if src in P.columns:
        P[f"XS_{src}"] = P.groupby("date")[src].rank(pct=True)
        FEAT.append(f"XS_{src}")

# ---- per-stock z-score: fit on TRAIN ROWS ONLY, apply to all ----
tr = P[P.split == "train"]
mu = tr.groupby("symbol")[FEAT].mean()
sd = tr.groupby("symbol")[FEAT].std().replace(0, np.nan)
mu_b = P[["symbol"]].join(mu, on="symbol")[FEAT]
sd_b = P[["symbol"]].join(sd, on="symbol")[FEAT]
P[FEAT] = ((P[FEAT].values - mu_b.values) / sd_b.values).astype(np.float32)
P[FEAT] = P[FEAT].clip(-8, 8)               # winsorize z-extremes

stats = (mu.add_suffix("__mu").join(sd.add_suffix("__sd")).reset_index())
os.makedirs(os.path.dirname(OUTF), exist_ok=True)
stats.to_parquet(OUTN, index=False)
for c in TGT:
    P[c] = P[c].astype(np.float32)
P.to_parquet(OUTF, index=False, compression="zstd")

print(f"[feat] wrote {OUTF}: {len(P):,} rows x {P.shape[1]} cols, "
      f"{P.symbol.nunique()} symbols, #features={len(FEAT)} "
      f"in {time.time()-t0:,.0f}s", flush=True)
print("[feat] split rows:",
      P.split.value_counts().to_dict(), flush=True)
print("[feat] norm stats fit on TRAIN only ->", OUTN,
      f"({stats.symbol.nunique()} symbols)", flush=True)
