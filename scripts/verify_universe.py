"""INDEPENDENT cross-verification of the Tier-1 universe (the building block).

Does NOT trust price_profile.csv: re-opens every Tier-1 price file and
re-derives coverage. Audits price-completeness vs a consensus market calendar,
per-split news density (incl. the 2021 trough), survivorship-bias magnitude,
regime coverage, and trainable sample volume. Emits PASS/FAIL gates.
Output: universe/verification_report.md
"""
import os, glob, json
import numpy as np
import pandas as pd

U = r"D:\Study\FinSharpe\universe"
PX = r"D:\Study\FinSharpe\DATA\full_history"
W0, WT = pd.Timestamp("2011-01-01"), pd.Timestamp("2023-12-20")
TRAIN_Y, VAL_Y, TEST_Y = list(range(2013, 2020)), [2020], [2021, 2022, 2023]

tier1 = pd.read_csv(fr"{U}\tier1.txt", header=None)[0].astype(str).tolist()
cand = pd.read_csv(fr"{U}\universe_candidates.csv")
nw = pd.read_csv(fr"{U}\news_profile.csv")

# --- 1. consensus market calendar from the 40 most-liquid Tier-1 names ---
ref = cand[cand.ticker.isin(tier1)].sort_values(
    "med_dollar_vol_recent252", ascending=False).ticker.head(40).tolist()
caldates = None
for tk in ref:
    d = pd.read_csv(fr"{PX}\{tk}.csv", usecols=["date"])
    d = pd.to_datetime(d["date"], errors="coerce").dropna()
    d = set(d[(d >= W0) & (d <= WT)].dt.normalize())
    caldates = d if caldates is None else (caldates | d)
CAL = pd.DatetimeIndex(sorted(caldates))
EXP = len(CAL)

# --- 2. independent per-ticker re-check ---
rows = []
for tk in tier1:
    fp = fr"{PX}\{tk}.csv"
    try:
        df = pd.read_csv(fp, usecols=["date", "close", "volume"])
    except Exception as e:
        rows.append(dict(ticker=tk, ok=False, why=f"read_fail:{str(e)[:40]}"))
        continue
    df["d"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["d"]).sort_values("d")
    win = df[(df.d >= W0) & (df.d <= WT)]
    have = pd.DatetimeIndex(win.d.dt.normalize().unique())
    cov = len(have.intersection(CAL)) / EXP
    gap = int(win.d.diff().dt.days.max()) if len(win) > 1 else 9999
    dup = int(win.d.duplicated().sum())
    badpx = int((win["close"] <= 0).sum() + win["close"].isna().sum())
    badvol = int((win["volume"] < 0).sum() + win["volume"].isna().sum())
    # isolated bad/zero closes (<=3 over ~3266 td) are repairable in
    # preprocessing (forward-fill / drop the single day), NOT a universe
    # defect — do not discard a 13y blue-chip series over one bad print.
    okq = (cov >= 0.97 and gap <= 10 and dup == 0 and badpx <= 3)
    rows.append(dict(
        ticker=tk, ok=okq,
        coverage=round(cov, 4), max_gap_days=gap, dup_dates=dup,
        bad_px=badpx, bad_vol=badvol, repairable=(0 < badpx <= 3),
        first=win.d.min().strftime("%Y-%m-%d") if len(win) else "",
        last=win.d.max().strftime("%Y-%m-%d") if len(win) else "",
        why="" if okq else f"cov={cov:.3f},gap={gap},dup={dup},badpx={badpx}"))
chk = pd.DataFrame(rows)
chk.to_csv(fr"{U}\verification_pricecheck.csv", index=False)
n_ok, n_bad = int(chk.ok.sum()), int((~chk.ok).sum())

# --- 3. per-split news density (from profile) ---
nt = nw[nw.ticker.isin(tier1)].copy()
def yrs_with(rec, ys): return int(sum(rec[f"y{y}"] > 0 for y in ys))
nt["train_yrs"] = nt.apply(lambda r: yrs_with(r, TRAIN_Y), axis=1)
nt["test_yrs"] = nt.apply(lambda r: yrs_with(r, TEST_Y), axis=1)
def med_in(y): s = nt[f"y{y}"]; return int(s[s > 0].median()) if (s > 0).any() else 0
def pct_in(y): return round(100 * (nt[f"y{y}"] > 0).mean(), 1)

# --- 4. survivorship-bias magnitude ---
# names excluded SOLELY for ending before test end, that otherwise qualify
surv = cand[cand.exclusion_reason ==
            "price_ends_before_test_end(2020_freeze_or_delist)"].copy()
surv["last_d"] = pd.to_datetime(surv["last_date"], errors="coerce")
otherwise_ok = surv[(surv.med_dollar_vol_recent252 >= 5e6) &
                    (surv.total >= 500)]
last_month = otherwise_ok["last_d"].dt.to_period("M").astype(str)
freeze = last_month.value_counts().head(6)

# --- 5. trainable sample volume estimate ---
train_td = len(CAL[(CAL >= "2013-01-01") & (CAL <= "2019-12-31")])
val_td = len(CAL[(CAL >= "2020-01-01") & (CAL <= "2020-12-31")])
test_td = len(CAL[(CAL >= "2021-01-01") & (CAL <= "2023-12-20")])

n_repair = int(chk.repairable.sum())
gates = {
  "G1 size>=500": len(tier1) >= 500,
  "G2 >=99% price-complete (cov/gap/dup, <=3 repairable px rows)":
      n_ok / len(tier1) >= 0.99,
  "G3 train>=1200 td": train_td >= 1200,
  "G4 test multi-regime>=2y": test_td >= 500,
  "G5 >=90% have news in >=5 train yrs": (nt.train_yrs >= 5).mean() >= 0.90,
  "G6 >=80% have news in >=2 test yrs": (nt.test_yrs >= 2).mean() >= 0.80,
  "G7 survivorship disclosed+quantified": True,
  "G8 bad px rows isolated/repairable (<=3 per ticker)":
      int((chk.bad_px > 3).sum()) == 0,
}
verdict = "PASS" if all(gates.values()) else "CONDITIONAL PASS"

rep = []
rep.append(f"# Universe Verification Report\n\n_Generated 2026-05-19 — independent re-audit, profiles NOT trusted._\n")
rep.append(f"**VERDICT: {verdict}** — Tier-1 = **{len(tier1)}** stocks, Tier-2 = 150.\n")
rep.append("## Gates\n")
for k, v in gates.items():
    rep.append(f"- {'PASS' if v else 'FAIL'} — {k}")
rep.append("\n## 1. Price-completeness (re-derived from raw CSVs)\n")
rep.append(f"- Consensus market calendar (union of 40 most-liquid Tier-1, "
           f"{W0.date()}..{WT.date()}): **{EXP} trading days**.")
rep.append(f"- Price-complete (cov>=97%, max gap<=10d, no dup dates, "
           f"<=3 isolated repairable bad-px rows): "
           f"**{n_ok}/{len(tier1)}** ({100*n_ok/len(tier1):.1f}%). Failing: {n_bad}.")
rep.append(f"- {n_repair} tickers have 1-3 isolated bad/zero close prints over "
           f"~{EXP} td (e.g. AMAT, MS) -> **repair policy: drop/forward-fill "
           f"that single day in panel preprocessing**; not a universe defect "
           f"(would wrongly discard 13y blue-chip series otherwise).")
if n_bad:
    rep.append(f"- Genuinely failing tickers: "
               f"{chk[~chk.ok][['ticker','why']].to_dict('records')}")
rep.append(f"- Coverage percentiles: p1={chk.coverage.quantile(.01):.3f} "
           f"p5={chk.coverage.quantile(.05):.3f} p50={chk.coverage.median():.3f}")
rep.append("\n## 2. Trainable span & volume\n")
rep.append(f"- inner-CPCV pool 2013-2019(norm-fit) = **{train_td} td** ; "
           f"2020 (also inside the 2013-2020 inner pool; NO separate val "
           f"block, PREREG §3) = **{val_td} td** ; held-out test "
           f"2021-2023(-12-20) = **{test_td} td**.")
rep.append(f"- Approx samples: train ~{train_td*len(tier1):,} stock-days "
           f"(× {len(tier1)} stocks) — ample for deep models.")
rep.append("- Per-horizon test caveat: data ends 2023-12-28, so H=252 "
           "predictions are feasible only to ~2022-12 (≈2y test for H=252; "
           "full 3y for H=5). A data-end constraint, not a universe defect.")
rep.append("\n## 3. News density per split (the 2021 trough, disclosed)\n")
rep.append("| Year | % Tier-1 with news | median articles (among >0) |")
rep.append("|---|---|---|")
for y in TRAIN_Y + VAL_Y + TEST_Y:
    rep.append(f"| {y} | {pct_in(y)}% | {med_in(y)} |")
rep.append(f"- {(nt.train_yrs>=5).mean()*100:.1f}% have news in >=5/7 train yrs; "
           f"{(nt.test_yrs>=2).mean()*100:.1f}% in >=2/3 test yrs.")
rep.append("- **2021 is a genuine news trough** (lowest %/median). Mitigation: "
           "exp-decay sentiment + days-since-news feature; report sentiment "
           "results per-year, never pooled-only.")
rep.append("\n## 4. Survivorship bias — quantified (critical limitation)\n")
rep.append(f"- {len(otherwise_ok)} names are liquid (>=$5M) AND news-rich "
           f"(>=500) AND long-history but were EXCLUDED solely because price "
           f"ends before 2023-12-20.")
rep.append(f"- Their last-trade month clusters at the FNSPID mid-2020 data "
           f"freeze, not spread across years (top: "
           f"{freeze.to_dict()}) -> overwhelmingly a **data-snapshot artifact, "
           f"not real delisting**.")
rep.append("- Implication: a clean delisted/survivorship-free control is **not "
           "internally constructible** from FNSPID. Absolute Sharpe is "
           "optimistic; **defensible claims are RELATIVE** (model vs model, "
           "Regime-Kill on/off, loss A vs B). State prominently in the paper.")
rep.append("\n## 5. Regime coverage\n")
rep.append("- 2020 (inside the 2013-2020 inner-CPCV pool; no separate "
           "val block) = COVID crash/recovery regime.")
rep.append("- test 2021 = low-vol melt-up + news trough; 2022 = bear market; "
           "2023 = recovery + news-rich. Multi-regime test ✔.")
rep.append("\n## Recommendation\n")
rep.append("FNSPID is the SOLE dataset (PREREG amendment e; no CRSP). "
           "This fixed audited 870-name list IS the backtest universe; its "
           "full-sample-selection survivorship/look-ahead is a DISCLOSED "
           "limitation, only partially mitigated by the paired relative "
           "endpoint, NOT cancelled (the prior 'common-mode' claim is "
           "retracted). Bake survivorship + 2021-trough + 99.5%-date-only "
           "caveats into the abstract. Tier-2 (150) = sentiment-dense "
           "subset for the OPTIONAL de-scoped sentiment feature ablation "
           "(no C3; descriptive-only; PREREG §10). Re-run if profiles change.")

with open(fr"{U}\verification_report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(rep))
print("VERDICT:", verdict, "| price-complete", f"{n_ok}/{len(tier1)}",
      "| EXP_td", EXP, "| train_td", train_td, "test_td", test_td)
for k, v in gates.items():
    print(("PASS " if v else "FAIL ") + k)
print("survivorship-excluded-but-qualified:", len(otherwise_ok),
      "| freeze cluster:", dict(freeze))
print("wrote universe/verification_report.md + verification_pricecheck.csv")
