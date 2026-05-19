"""Per-stock daily exp-decayed sentiment feature (DE-SCOPED to a feature).

Re-jury Jury5 FATAL-1: FNSPID `Date` is ~99.5% date-only (00:00:00 UTC;
2009-2018 = 100% date-only). There is NO usable intraday resolution, so
UTC->ET 'pre-open' logic was theatre on a date field (and mildly
anti-conservative). Corrected, genuinely no-look-ahead rule:

  effective trading session = first trading day STRICTLY AFTER the stated
  calendar date D (D = date part of `ts`). A news item stamped date D is
  actionable only at the next session — independent of any (absent)
  intraday time. The ~0.5% real-timestamp rows are not special-cased
  (uniform conservative rule); an intraday-resolved arm is future work.

Built from universe/tier1_news_scored.parquet (`ts` carries the date).
Sentiment is an OPTIONAL feature ablation only (PREREG §10), not a claim.

Output: panel/sentiment_daily.parquet
[symbol,date,sent_decay,sent_raw,news_intensity,sent_dispersion,
 days_since_news,has_news]
"""
import argparse, os, time
import numpy as np
import pandas as pd

AP = argparse.ArgumentParser()
AP.add_argument("--half_life", type=int, default=5)
AP.add_argument("--win_start", default="2011-01-01")
AP.add_argument("--win_end", default="2023-12-28")
A = AP.parse_args()

SC = r"D:\Study\FinSharpe\universe\tier1_news_scored.parquet"
PXDIR = r"D:\Study\FinSharpe\DATA\full_history"
TIER1 = r"D:\Study\FinSharpe\universe\tier1.txt"
OUT = r"D:\Study\FinSharpe\panel\sentiment_daily.parquet"
LAM = 0.5 ** (1.0 / A.half_life)
W0, W1 = pd.Timestamp(A.win_start), pd.Timestamp(A.win_end)

t0 = time.time()
tier1 = pd.read_csv(TIER1, header=None)[0].astype(str).str.strip().tolist()
sc = pd.read_parquet(SC, columns=["symbol", "ts", "sent_signed"])
# stated calendar date = first 10 chars of the FNSPID Date string. NO tz
# conversion (the field is a date; 99.5% are 00:00:00 UTC).
sc["sdate"] = pd.to_datetime(sc["ts"].str.slice(0, 10), errors="coerce")
sc = sc.dropna(subset=["sdate"])
date_only_frac = float((pd.to_datetime(sc["ts"], utc=True, errors="coerce")
                        .dt.strftime("%H:%M:%S") == "00:00:00").mean())
g = {s: d for s, d in sc.groupby("symbol", sort=False)}
print(f"[sent] {len(tier1)} symbols lambda={LAM:.4f} (H={A.half_life}); "
      f"scored={len(sc):,}; ts date-only frac={date_only_frac:.4f} "
      f"(disclosed: FNSPID Date has no intraday resolution)", flush=True)

parts, miss, la_viol = [], [], 0
for i, sym in enumerate(tier1):
    fp = os.path.join(PXDIR, f"{sym}.csv")
    if not os.path.exists(fp):
        miss.append(sym); continue
    td = pd.read_csv(fp, usecols=["date"])
    td = pd.to_datetime(td["date"], errors="coerce").dropna()
    td = np.sort(td[(td >= W0) & (td <= W1)].dt.normalize().unique())
    N = len(td)
    if N == 0:
        miss.append(sym); continue

    raw = np.full(N, np.nan); cnt = np.zeros(N, np.int32)
    disp = np.full(N, np.nan)
    a = g.get(sym)
    if a is not None and len(a):
        D = a["sdate"].values.astype("datetime64[ns]")
        # first trading day STRICTLY AFTER the stated date (conservative)
        eff = np.searchsorted(td, D, side="right")
        m = eff < N
        if m.any():
            ei = eff[m]; sv = a["sent_signed"].values[m]
            # conservative no-look-ahead: effective session date MUST be
            # strictly greater than the stated news date.
            la_viol += int((td[ei] <= D[m]).sum())
            tmp = pd.DataFrame({"ei": ei, "s": sv})
            ag = tmp.groupby("ei")["s"].agg(["mean", "count", "std"])
            raw[ag.index.values] = ag["mean"].values
            cnt[ag.index.values] = ag["count"].values.astype(np.int32)
            disp[ag.index.values] = np.nan_to_num(ag["std"].values, nan=0.0)

    news = ~np.isnan(raw)
    idx = np.arange(N)
    last_idx = pd.Series(np.where(news, idx, np.nan)).ffill().values
    last_raw = pd.Series(np.where(news, raw, np.nan)).ffill().values
    has_prev = ~np.isnan(last_idx)
    gap = np.where(has_prev, idx - np.nan_to_num(last_idx), -1).astype(np.int64)
    decay = np.where(has_prev, np.nan_to_num(last_raw) * (LAM ** gap), 0.0)

    parts.append(pd.DataFrame({
        "symbol": sym, "date": td,
        "sent_decay": decay.astype(np.float32),
        "sent_raw": raw.astype(np.float32),
        "news_intensity": cnt,
        "sent_dispersion": np.where(news, disp, np.nan).astype(np.float32),
        "days_since_news": np.where(has_prev, gap, -1).astype(np.int64),
        "has_news": news}))
    if (i + 1) % 200 == 0:
        print(f"[sent] {i+1}/{len(tier1)} {time.time()-t0:,.0f}s", flush=True)

panel = pd.concat(parts, ignore_index=True)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
panel.to_parquet(OUT, index=False, compression="zstd")
print(f"[sent] wrote {OUT}: {len(panel):,} rows, {panel.symbol.nunique()} "
      f"symbols, {panel.date.min().date()}..{panel.date.max().date()} "
      f"in {time.time()-t0:,.0f}s", flush=True)
print(f"[sent] conservative-rule look-ahead violations: {la_viol} "
      f"(MUST be 0); missing-price: {len(miss)}", flush=True)
print(f"[sent] has_news days: {panel.has_news.mean()*100:.1f}% ; "
      f"sent_decay mean={panel.sent_decay.mean():.4f}", flush=True)
