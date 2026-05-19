# Universe Verification Report

_Generated 2026-05-19 — independent re-audit, profiles NOT trusted._

**VERDICT: PASS** — Tier-1 = **870** stocks, Tier-2 = 150.

## Gates

- PASS — G1 size>=500
- PASS — G2 >=99% price-complete (cov/gap/dup, <=3 repairable px rows)
- PASS — G3 train>=1200 td
- PASS — G4 test multi-regime>=2y
- PASS — G5 >=90% have news in >=5 train yrs
- PASS — G6 >=80% have news in >=2 test yrs
- PASS — G7 survivorship disclosed+quantified
- PASS — G8 bad px rows isolated/repairable (<=3 per ticker)

## 1. Price-completeness (re-derived from raw CSVs)

- Consensus market calendar (union of 40 most-liquid Tier-1, 2011-01-01..2023-12-20): **3266 trading days**.
- Price-complete (cov>=97%, max gap<=10d, no dup dates, <=3 isolated repairable bad-px rows): **870/870** (100.0%). Failing: 0.
- 17 tickers have 1-3 isolated bad/zero close prints over ~3266 td (e.g. AMAT, MS) -> **repair policy: drop/forward-fill that single day in panel preprocessing**; not a universe defect (would wrongly discard 13y blue-chip series otherwise).
- Coverage percentiles: p1=0.999 p5=0.999 p50=0.999

## 2. Trainable span & volume

- inner-CPCV pool 2013-2019(norm-fit) = **1763 td** ; 2020 (also inside the 2013-2020 inner pool; NO separate val block, PREREG §3) = **253 td** ; held-out test 2021-2023(-12-20) = **747 td**.
- Approx samples: train ~1,533,810 stock-days (× 870 stocks) — ample for deep models.
- Per-horizon test caveat: data ends 2023-12-28, so H=252 predictions are feasible only to ~2022-12 (≈2y test for H=252; full 3y for H=5). A data-end constraint, not a universe defect.

## 3. News density per split (the 2021 trough, disclosed)

| Year | % Tier-1 with news | median articles (among >0) |
|---|---|---|
| 2013 | 97.8% | 126 |
| 2014 | 99.0% | 136 |
| 2015 | 100.0% | 204 |
| 2016 | 100.0% | 191 |
| 2017 | 100.0% | 195 |
| 2018 | 100.0% | 300 |
| 2019 | 100.0% | 321 |
| 2020 | 100.0% | 155 |
| 2021 | 98.5% | 50 |
| 2022 | 99.9% | 93 |
| 2023 | 100.0% | 113 |
- 100.0% have news in >=5/7 train yrs; 100.0% in >=2/3 test yrs.
- **2021 is a genuine news trough** (lowest %/median). Mitigation: exp-decay sentiment + days-since-news feature; report sentiment results per-year, never pooled-only.

## 4. Survivorship bias — quantified (critical limitation)

- 338 names are liquid (>=$5M) AND news-rich (>=500) AND long-history but were EXCLUDED solely because price ends before 2023-12-20.
- Their last-trade month clusters at the FNSPID mid-2020 data freeze, not spread across years (top: {'2020-07': 208, '2020-06': 72, '2020-04': 56, '2023-11': 2}) -> overwhelmingly a **data-snapshot artifact, not real delisting**.
- Implication: a clean delisted/survivorship-free control is **not internally constructible** from FNSPID. Absolute Sharpe is optimistic; **defensible claims are RELATIVE** (model vs model, Regime-Kill on/off, loss A vs B). State prominently in the paper.

## 5. Regime coverage

- 2020 (inside the 2013-2020 inner-CPCV pool; no separate val block) = COVID crash/recovery regime.
- test 2021 = low-vol melt-up + news trough; 2022 = bear market; 2023 = recovery + news-rich. Multi-regime test ✔.

## Recommendation

FNSPID is the SOLE dataset (PREREG amendment e; no CRSP). This fixed audited 870-name list IS the backtest universe; its full-sample-selection survivorship/look-ahead is a DISCLOSED limitation, only partially mitigated by the paired relative endpoint, NOT cancelled (the prior 'common-mode' claim is retracted). Bake survivorship + 2021-trough + 99.5%-date-only caveats into the abstract. Tier-2 (150) = sentiment-dense subset for the OPTIONAL de-scoped sentiment feature ablation (no C3; descriptive-only; PREREG §10). Re-run if profiles change.