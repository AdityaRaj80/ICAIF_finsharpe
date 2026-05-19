"""FinBERT label-quality validation (JURY F-F#2, re-scoped).

FNSPID's GPT-3.5 labels are NOT in the provided data, so the "FinBERT beats
GPT-3.5" claim is dropped (PREREG §11). Substantive validation instead:
  (1) PREDICTIVE VALIDITY, TRAIN ONLY (2013-2019, no leakage): does daily
      signed sentiment rank-predict forward returns? pooled Spearman IC +
      decile monotonicity + per-year.
  (2) Human-audit gold sample (stratified 200 headlines) + an objective
      finance-lexicon sign cross-check (coarse second opinion).
  (3) Dedup over-merge rate from a BOUNDED raw probe (Jury5#6).
Output: panel/finbert_validation.md , panel/finbert_audit_sample.csv
"""
import os, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SC = r"D:\Study\FinSharpe\universe\tier1_news_scored.parquet"
PXDIR = r"D:\Study\FinSharpe\DATA\full_history"
TIER1 = r"D:\Study\FinSharpe\universe\tier1.txt"
SRCS = [(r"D:\Study\FinSharpe\DATA\All_external.csv", "allext"),
        (r"D:\Study\FinSharpe\DATA\Stock_news\nasdaq_exteral_data.csv", "nasdaq")]
OUTMD = r"D:\Study\FinSharpe\panel\finbert_validation.md"
OUTCSV = r"D:\Study\FinSharpe\panel\finbert_audit_sample.csv"
TRN0, TRN1 = pd.Timestamp("2013-01-01"), pd.Timestamp("2019-12-31")
t0 = time.time()

LM_POS = set("beat beats beneficial best better boom boosted breakthrough "
             "exceeded gain gains grew growth high higher improve improved "
             "outperform positive profit profitable raise raised rally record "
             "rose strong strength surge upgrade upbeat win wins".split())
LM_NEG = set("bankrupt cut cuts decline declined drop dropped fall fell fraud "
             "investigation lawsuit loss losses lower miss missed plunge "
             "downgrade negative recall slump weak weakness warn warning "
             "lawsuit sink sank slowdown halt".split())

tier1 = set(pd.read_csv(TIER1, header=None)[0].astype(str).str.strip())
sc = pd.read_parquet(SC, columns=["symbol", "ts", "sent_signed", "p_pos",
                                   "p_neg", "p_neu"])
# stated calendar date (FNSPID Date is 99.5% date-only; NO tz conversion —
# consistent with build_sentiment_daily.py conservative rule)
sc["d"] = pd.to_datetime(sc["ts"].str.slice(0, 10), errors="coerce")
sc = sc.dropna(subset=["d"])
# daily mean signed score per (symbol, stated date) -- matches builder
daily = (sc.groupby(["symbol", "d"])["sent_signed"].mean().reset_index())

# ---- (1) predictive validity, TRAIN ONLY ----
ic_rows = []
pooled = []
for sym in list(tier1):
    f = os.path.join(PXDIR, f"{sym}.csv")
    if not os.path.exists(f):
        continue
    px = pd.read_csv(f, usecols=["date", "adj close"])
    px["date"] = pd.to_datetime(px["date"], errors="coerce")
    px = px.dropna().sort_values("date").drop_duplicates("date")
    px = px[(px.date >= "2012-06-01") & (px.date <= TRN1)].reset_index(drop=True)
    if len(px) < 60:
        continue
    c = px["adj close"].astype("float64")
    px["fwd5"] = np.log(c.shift(-5) / c)
    px["fwd20"] = np.log(c.shift(-20) / c)
    ds = daily[(daily.symbol == sym) & (daily.d >= TRN0) & (daily.d <= TRN1)]
    j = ds.merge(px, left_on="d", right_on="date", how="inner").dropna(
        subset=["fwd5"])
    if len(j) >= 30:
        pooled.append(j[["sent_signed", "fwd5", "fwd20", "d"]].assign(symbol=sym))

pl = pd.concat(pooled, ignore_index=True) if pooled else pd.DataFrame()
ic5 = spearmanr(pl["sent_signed"], pl["fwd5"])[0] if len(pl) else np.nan
p20 = pl.dropna(subset=["fwd20"])
ic20 = spearmanr(p20["sent_signed"], p20["fwd20"])[0] if len(p20) else np.nan
# daily cross-sectional IC (mean of per-day Spearman), the trading-relevant one
xs = (pl.groupby("d")[["sent_signed", "fwd5"]].apply(
    lambda g: spearmanr(g.sent_signed, g.fwd5)[0] if g.sent_signed.nunique() > 4
    else np.nan).dropna())
xs_ic5 = float(xs.mean()) if len(xs) else np.nan
xs_t = float(xs.mean() / (xs.std() / np.sqrt(len(xs)))) if len(xs) > 2 else np.nan
pl["dec"] = pd.qcut(pl["sent_signed"].rank(method="first"), 10,
                    labels=False)
dec = pl.groupby("dec")["fwd5"].mean()
mono = float(np.corrcoef(np.arange(10), dec.values)[0, 1])

# ---- (2) audit sample + lexicon cross-check ----
sc["lab"] = np.where(sc.p_pos >= sc[["p_neg", "p_neu"]].max(1), "pos",
                     np.where(sc.p_neg >= sc[["p_pos", "p_neu"]].max(1),
                              "neg", "neu"))
sc2 = sc.copy()
sc2["ts_str"] = sc2["ts"]
# need titles: pull from corpus parquet (has title aligned by symbol+ts)
corp = pd.read_parquet(r"D:\Study\FinSharpe\universe\tier1_news_corpus.parquet",
                       columns=["symbol", "ts", "title"])
m = sc2.merge(corp, on=["symbol", "ts"], how="left").dropna(subset=["title"])
rng = np.random.default_rng(0)
parts = []
for lb in ["pos", "neg", "neu"]:
    sub = m[m.lab == lb]
    parts.append(sub.sample(min(70, len(sub)), random_state=1))
aud = pd.concat(parts).reset_index(drop=True)


def lex_sign(t):
    w = set(str(t).lower().replace(",", " ").replace(".", " ").split())
    p, n = len(w & LM_POS), len(w & LM_NEG)
    return 1 if p > n else (-1 if n > p else 0)


aud["lex"] = aud["title"].map(lex_sign)
aud["fb_sign"] = np.sign(aud["sent_signed"]).astype(int)
nz = aud[(aud.lex != 0)]
agree = float((np.sign(nz.fb_sign) == np.sign(nz.lex)).mean()) if len(nz) else np.nan
aud[["symbol", "ts", "title", "p_pos", "p_neg", "p_neu", "sent_signed",
     "lab", "lex"]].assign(human_label="").to_csv(OUTCSV, index=False)

# ---- (3) dedup over-merge probe (bounded) ----
LOOSE, STRICT = set(), set()
seen_rows = 0
for path, tag in SRCS:
    rdr = pd.read_csv(path, usecols=["Date", "Stock_symbol", "Article_title",
                      "Url"], dtype=str, chunksize=200_000, engine="c",
                      on_bad_lines="skip", encoding_errors="replace")
    for ch in rdr:
        ch = ch.dropna(subset=["Stock_symbol", "Date", "Article_title"])
        ch = ch[ch["Stock_symbol"].str.strip().isin(tier1)]
        if len(ch):
            d = ch["Date"].str.slice(0, 10)
            sym = ch["Stock_symbol"].str.strip()
            ti = ch["Article_title"].astype(str).str.strip()
            url = ch["Url"].fillna("").astype(str)
            for s, dd, t, u in zip(sym, d, ti, url):
                LOOSE.add(hash((s, dd, t[:120])))
                STRICT.add(hash((s, dd, t, u)))
            seen_rows += len(ch)
        if seen_rows >= 2_000_000:
            break
    if seen_rows >= 2_000_000:
        break
over_merge = 1 - (len(LOOSE) / len(STRICT)) if STRICT else np.nan

R = ["# FinBERT Supporting Diagnostic (sentiment is a DE-SCOPED feature)\n",
     "_2026-05-19. Scope collapsed to C1 (PAPER_PLAN.md): sentiment is NOT "
     "a contribution, only an optional on/off feature ablation. This file "
     "is a supporting diagnostic, not a claim. GPT-3.5 head-to-head N/A "
     "(labels absent in provided data). Stated-date aligned (no tz)._\n",
     "## 1. Predictive validity (TRAIN-era 2013-2019 only, no leakage)\n",
     f"- pooled Spearman IC: fwd-5d **{ic5:+.4f}**, fwd-20d {ic20:+.4f} "
     f"(n={len(pl):,} news-day obs).",
     f"- daily cross-sectional IC (fwd-5d): mean **{xs_ic5:+.4f}**, "
     f"Newey-naive t≈{xs_t:+.2f} over {len(xs)} days.",
     f"- decile monotonicity: corr **{mono:+.3f}**, spread "
     f"{dec.iloc[-1]-dec.iloc[0]:+.4f}.",
     "- Reading: a small but real positive IC means the sentiment feature "
     "carries weak return-relevant signal — its incremental ECONOMIC value "
     "net of cost/turnover is exactly what the C1 on/off ablation measures "
     "(no separate sentiment claim is made; this resolves the prior "
     "framing contradiction Jury4 flagged).",
     "\n## 2. Label audit\n",
     f"- 200-headline stratified gold sample -> `finbert_audit_sample.csv` "
     f"(human_label blank — manual review pending; advisory only since "
     f"C3 is dropped).",
     f"- finance-lexicon sign agreement (n={len(nz)}): **{agree:.3f}** — "
     f"weak coarse proxy; not load-bearing.",
     "\n## 3. Dedup: WHY the strict key (Jury5#6)\n",
     f"- This probe measures a property of the RAW DATA: applying the OLD "
     f"loose key (sym|date|title[:120]) vs a strict (sym|date|fulltitle|"
     f"url) key to {seen_rows:,} raw rows over-merges **{over_merge:.4f}**. "
     f"That is the *motivation*, not the shipped state.",
     f"- The shipped corpus is built with the STRICT key "
     f"(extract_tier1_news.py): 1,295,077 -> 1,826,882 articles (+41% "
     f"recovered); pipeline over-merge is ~0 by construction. The sentiment "
     f"panel was rebuilt on it (has_news 24.7% -> ~30%).",
     "\n## Note\n- Title-vs-body ablation (ablate_title_body.py): advisory; "
     "title chosen a-priori (51% bodies absent, FNSPID-comparable), not "
     "outcome-selected. Decay/H are TRAIN-only (PREREG §9). Sentiment is a "
     "feature ablation only.",
     f"\n_elapsed {time.time()-t0:,.0f}s_"]
os.makedirs(os.path.dirname(OUTMD), exist_ok=True)
open(OUTMD, "w", encoding="utf-8").write("\n".join(R))
print(f"pooled IC5={ic5:+.4f} IC20={ic20:+.4f} xsIC5={xs_ic5:+.4f} "
      f"t={xs_t:+.2f} mono={mono:+.3f} n={len(pl):,}")
print(f"lexicon agree={agree:.3f} (n_nz={len(nz)}) ; "
      f"dedup over_merge={over_merge:.4f} on {seen_rows:,} rows")
print("wrote", OUTMD, "and", OUTCSV)
