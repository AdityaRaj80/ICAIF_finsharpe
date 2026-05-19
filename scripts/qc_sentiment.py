"""INDEPENDENT QC of the sentiment panel (conservative date-only rule).

Re-jury Jury5 fix: (a) the QC MEASURES and REPORTS the fraction of `ts`
that is date-only (00:00:00) so the date-only nature is disclosed, not
hidden; (b) re-derives the effective session straight from raw `ts` stated
date WITHOUT reading the builder, then G6 checks set-equality vs the
builder; (c) the no-look-ahead gate is the genuinely-conservative one:
effective session date STRICTLY AFTER the stated news date.
Output: panel/sentiment_qc_report.md
"""
import os, time
import numpy as np
import pandas as pd

SC = r"D:\Study\FinSharpe\universe\tier1_news_scored.parquet"
PAN = r"D:\Study\FinSharpe\panel\sentiment_daily.parquet"
PXDIR = r"D:\Study\FinSharpe\DATA\full_history"
TIER1 = r"D:\Study\FinSharpe\universe\tier1.txt"
OUT = r"D:\Study\FinSharpe\panel\sentiment_qc_report.md"
H = 5
LAM = 0.5 ** (1.0 / H)
W0, W1 = pd.Timestamp("2011-01-01"), pd.Timestamp("2023-12-28")
t0 = time.time()

tier1 = pd.read_csv(TIER1, header=None)[0].astype(str).str.strip().tolist()
pan = pd.read_parquet(PAN)
pan["y"] = pan["date"].dt.year

sc = pd.read_parquet(SC, columns=["symbol", "ts", "sent_signed"])
sc["sdate"] = pd.to_datetime(sc["ts"].str.slice(0, 10), errors="coerce")
sc = sc.dropna(subset=["sdate"])
tod = pd.to_datetime(sc["ts"], utc=True, errors="coerce").dt.strftime(
    "%H:%M:%S")
date_only_frac = float((tod == "00:00:00").mean())
yr_dof = (pd.DataFrame({"y": sc["sdate"].dt.year, "z": (tod == "00:00:00")})
          .groupby("y")["z"].mean())

rng = np.random.default_rng(0)
samp = list(rng.choice(tier1, size=min(150, len(tier1)), replace=False))
la_viol = mismatch = 0
for sym in samp:
    fp = os.path.join(PXDIR, f"{sym}.csv")
    td = pd.read_csv(fp, usecols=["date"])
    td = pd.to_datetime(td["date"], errors="coerce").dropna()
    td = np.sort(td[(td >= W0) & (td <= W1)].dt.normalize().unique())
    a = sc[sc.symbol == sym]
    if not len(a) or not len(td):
        continue
    D = a["sdate"].values.astype("datetime64[ns]")
    eff = np.searchsorted(td, D, side="right")        # strictly AFTER D
    m = eff < len(td)
    ed = td[eff[m]]
    la_viol += int((ed <= D[m]).sum())                # must be 0
    indep = set(pd.Series(ed).dt.strftime("%Y-%m-%d"))
    built = set(pan[(pan.symbol == sym) & pan.has_news]["date"]
                .dt.strftime("%Y-%m-%d"))
    if indep != built:
        mismatch += 1

yrt = pan.groupby("y").agg(rows=("has_news", "size"),
                           news_pct=("has_news", lambda x: x.mean() * 100),
                           mi=("news_intensity", "mean")).round(2)
nrow = pan[pan.has_news]
g3a = bool((nrow.days_since_news == 0).all())
g3b = bool(np.allclose(nrow.sent_decay, nrow.sent_raw, atol=1e-5))
ch = pan.sort_values(["symbol", "date"]).copy()
ch["ad"] = ch.sent_decay.abs()
ch["pa"] = ch.groupby("symbol")["ad"].shift(1)
nn = ch[(~ch.has_news) & ch.pa.notna()]
g3c = bool((nn.ad <= nn.pa + 1e-6).all())
g4 = bool(pan.sent_decay.between(-1, 1).all() and pan.sent_decay.std() > 0)
mkt = (pan[pan.has_news].assign(m=pan.date.dt.to_period("M"))
       .groupby("m")["sent_raw"].mean())
covid = float(mkt.get(pd.Period("2020-03"), np.nan))
base = float(mkt.loc[[p for p in [pd.Period("2019-12"), pd.Period("2020-01"),
                                  pd.Period("2020-02")] if p in mkt.index]]
             .mean())

gates = {
 "G1 EXACTLY 0 look-ahead (effective session strictly AFTER stated date)":
     la_viol == 0,
 "G2 every year has news (>0); 2021 trough disclosed":
     bool((yrt.news_pct > 0).all()),
 "G3a news-day days_since_news==0 (exact)": g3a,
 "G3b news-day sent_decay==sent_raw (exact)": g3b,
 "G3c no-news decay monotone non-increasing (exact)": g3c,
 "G4 sent_decay in [-1,1], variance>0": g4,
 "G5 macro 2020-03 COVID dip vs Dec19-Feb20": covid < base,
 "G6 independent re-derivation == builder (0 mismatched syms)":
     mismatch == 0,
 "G7 ts date-only fraction MEASURED & disclosed (not gated, reported)":
     True,
}
verdict = "PASS" if all(gates.values()) else "FAIL"

R = [f"# Sentiment Panel QC (conservative date-only rule)\n",
     f"_2026-05-19. Sentiment is a DE-SCOPED optional feature (PREREG §10), "
     f"not a contribution. Rule: effective session = first trading day "
     f"STRICTLY AFTER the stated date (no intraday assumption)._\n",
     f"**VERDICT: {verdict}** — {len(pan):,} rows, {pan.symbol.nunique()} "
     f"symbols.\n",
     f"## CRITICAL DISCLOSURE — FNSPID timestamp resolution\n",
     f"- **{date_only_frac*100:.2f}%** of scored `ts` are `00:00:00 UTC` "
     f"(date-only). FNSPID `Date` has **no usable intraday resolution**; "
     f"any intraday/point-in-time precision claim is unsupported. The "
     f"conservative strict-next-session rule is therefore the only honest "
     f"alignment. Per-year date-only fraction:",
     "| year | date-only |", "|--|--|"]
for y, v in yr_dof.items():
    R.append(f"| {int(y)} | {v*100:.1f}% |")
R.append("\n## Gates\n")
for k, v in gates.items():
    R.append(f"- {'PASS' if v else 'FAIL'} — {k}")
R += [f"\n## Point-in-time ({len(samp)}-symbol independent re-derivation)\n"
      f"- look-ahead violations (effective <= stated date): **{la_viol}** "
      f"(must be 0).",
      f"- builder vs independent session-set mismatch: **{mismatch}** "
      f"(must be 0).",
      "\n## Per-year coverage\n", "| year | rows | %news | mean intensity |",
      "|--|--|--|--|"]
for y, r in yrt.iterrows():
    R.append(f"| {y} | {int(r.rows):,} | {r.news_pct} | {r.mi} |")
R += [f"\n## Decay / distribution / macro\n- dsn==0 {g3a}; decay==raw "
      f"{g3b}; monotone {g3c}; sent_decay mean "
      f"{pan.sent_decay.mean():.4f} std {pan.sent_decay.std():.4f}; "
      f"COVID 2020-03 {covid:.4f} vs base {base:.4f}.",
      "\n## Note\n- Prior UTC->ET 'pre-open' treatment WITHDRAWN (theatre "
      "on a date-only field; was mildly anti-conservative). Sentiment used "
      "only as an on/off feature ablation."]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w", encoding="utf-8").write("\n".join(R))
print("VERDICT:", verdict, f"({time.time()-t0:,.0f}s)")
for k, v in gates.items():
    print(("PASS " if v else "FAIL ") + k)
print(f"la_viol={la_viol} mismatch={mismatch} date_only={date_only_frac:.4f} "
      f"covid={covid:.4f} base={base:.4f}")
print("wrote", OUT)
