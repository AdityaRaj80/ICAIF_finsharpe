"""Profile every full_history price CSV.

Output: universe/price_profile.csv with one row per ticker:
  ticker, n_rows, first_date, last_date, n_years,
  med_dollar_vol_recent252, med_dollar_vol_full, max_gap_days, looks_like_etf

Price files: columns = date,volume,open,high,low,close,adj close ; newest-first.
Liquidity uses RAW close * volume (traded dollar value), not adj close.
"""
import os, glob, sys
import numpy as np
import pandas as pd

SRC = r"D:\Study\FinSharpe\DATA\full_history"
OUT = r"D:\Study\FinSharpe\universe\price_profile.csv"

files = sorted(glob.glob(os.path.join(SRC, "*.csv")))
print(f"[prices] {len(files)} files", flush=True)

rows = []
for i, fp in enumerate(files):
    tk = os.path.splitext(os.path.basename(fp))[0]
    try:
        df = pd.read_csv(fp, usecols=["date", "volume", "close"])
    except Exception as e:
        rows.append(dict(ticker=tk, n_rows=0, first_date="", last_date="",
                          n_years=0.0, med_dollar_vol_recent252=0.0,
                          med_dollar_vol_full=0.0, max_gap_days=-1,
                          error=str(e)[:80]))
        continue
    if df.empty:
        rows.append(dict(ticker=tk, n_rows=0, first_date="", last_date="",
                          n_years=0.0, med_dollar_vol_recent252=0.0,
                          med_dollar_vol_full=0.0, max_gap_days=-1, error="empty"))
        continue
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    n = len(df)
    if n == 0:
        rows.append(dict(ticker=tk, n_rows=0, first_date="", last_date="",
                          n_years=0.0, med_dollar_vol_recent252=0.0,
                          med_dollar_vol_full=0.0, max_gap_days=-1, error="no_valid_dates"))
        continue
    dv = (df["close"].astype("float64") * df["volume"].astype("float64"))
    first_d, last_d = df["date"].iloc[0], df["date"].iloc[-1]
    n_years = (last_d - first_d).days / 365.25
    # largest gap between consecutive trading rows, in calendar days
    gaps = df["date"].diff().dt.days.dropna()
    max_gap = int(gaps.max()) if len(gaps) else 0
    rows.append(dict(
        ticker=tk,
        n_rows=n,
        first_date=first_d.strftime("%Y-%m-%d"),
        last_date=last_d.strftime("%Y-%m-%d"),
        n_years=round(n_years, 2),
        med_dollar_vol_recent252=float(np.median(dv.iloc[-252:])),
        med_dollar_vol_full=float(np.median(dv)),
        max_gap_days=max_gap,
        error="",
    ))
    if (i + 1) % 1000 == 0:
        print(f"[prices] {i+1}/{len(files)}", flush=True)

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
print(f"[prices] wrote {OUT}  ({len(out)} tickers)", flush=True)
print(out[["n_rows", "n_years", "med_dollar_vol_recent252"]].describe().to_string(), flush=True)
