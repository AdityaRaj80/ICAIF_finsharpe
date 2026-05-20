# 01 — Dataset and universe

## The raw input: FNSPID

FNSPID (Financial News and Stock Price Integration Dataset, Dong et al.
2024) is a public US-equity dataset combining ~7,693 tickers'
daily-bar price history with two large news corpora
(`All_external.csv`, `nasdaq_exteral_data.csv`). Total raw size ≈
30 GB. Three structural problems were known going in and disclosed:

1. **2020 snapshot artifact** — the price coverage was pulled at a
   particular date; tickers that were delisted before then are
   under-represented and tickers that were delisted *after* the
   snapshot vanish. There is no clean delisted cohort to control
   survivorship.
2. **2021 news trough** — the news corpus thins materially in 2021,
   inflating any sentiment signal's apparent stability in that year.
3. **99.5% of news timestamps are date-only** — the raw news has no
   intraday timestamp for the vast majority of rows. Any "publish-
   minute aware" sentiment alignment is fiction.

These three are not bugs that can be fixed — they are properties of
the dataset that the project must *disclose and respect* rather than
hide.

## Stage 1: profiling

`scripts/profile_prices.py` and `scripts/profile_news.py` were run
against the unpacked FNSPID to produce `universe/price_profile.csv`
and `universe/news_profile.csv`. Per-ticker we measured: row count,
date range, max gap, ETF flag (curated list of broad ETFs to exclude),
and per-ticker news article count.

The distributions confirmed:
- A long tail of micro-coverage tickers (a few hundred price rows).
- A small head of mega-coverage tickers (mostly large-cap and ETFs).
- A bimodal news distribution: most tickers near zero news, a head of
  a few thousand tickers with substantive coverage.

`scripts/analyze_profiles.py` produced the cumulative coverage
trade-off curves that drove the universe specification.

## Stage 2: universe specification

The universe spec (frozen in `universe/universe_spec.json`) was
chosen BEFORE any modelling and is the file that pins the modeling
window and the inclusion gates.

Tier-1 inclusion (the universe used by the backtest):
- **price coverage**: continuous daily rows ≥ ~3,000 trading days,
  max gap ≤ 10 trading days, in the 2011-01-01 to 2023-12-31 window;
- **news coverage**: per-ticker news count above a threshold (chosen
  from the cumulative coverage curve to keep ~870 names);
- **liquidity**: rough proxy via average dollar volume in the window
  (no exotic micro-caps);
- **share class**: common shares only; curated ETF list excluded.

This produced **870 Tier-1 tickers** (the actual file:
`universe/tier1.txt`). Tier-2 (smaller) is kept for sensitivity but
not used in the main backtest.

## Stage 3: verification (re-jury 1)

`scripts/verify_universe.py` cross-checks the universe by
re-computing the inclusion gates from a different code path and
comparing. The audit report
(`universe/verification_report.md`) confirmed the 870-name list is
the deterministic image of the spec; `verification_pricecheck.csv`
spot-checked dates/coverage per ticker.

## Stage 4: sensitivity (the ±1-step band)

`scripts/sensitivity_universe.py` perturbed each inclusion threshold
by ±1 economic step (e.g., min-rows ±250, max-gap ±2, news threshold
±10%) and re-ran the spec. The `universe/sensitivity_report.md`
documents the resulting universe-size band. Headline: the universe is
stable to ±1-step perturbation in size; the relative-paired endpoint
is documented to additionally be robust because the same band is used
for both arms.

## Disclosed selection survivorship

The 870 list was **full-sample-selected** (completeness/liquidity/news
gates evaluated on the entire window). That is honest about being a
disclosed limitation, not a bug masked away:

- Names whose price history did not span the window are not in the
  universe — including names that *failed* during the window. The
  universe therefore over-represents winners (selection survivorship).
- The H1 paired contrast partially — **but only partially** — mitigates
  this, because both arms train and trade on the same universe.
  The residual concern is **survivorship-interaction bias**: two
  objectives select different sub-portfolios with different exposure
  to the (missing) delisted mass; the paired difference can be biased
  in a self-favoring direction. This is disclosed in PAPER_PLAN and is
  the reason every absolute level metric is appendix-only.

## What was discarded and why

- **Adding a second dataset (e.g., CRSP)**: the user does not have
  CRSP access at BITS. Considered alternatives (CMIE Prowess, etc.);
  none added the missing delisted cohort for US equities without a
  large engineering tail. Decision: FNSPID-only relative reframe
  (Task 33). The paper frames the work as a methodology re-analysis
  of one widely-used but contaminated benchmark, not a cross-dataset
  external-validity claim.
- **Including the 17 repairable-row tickers**: a small set of
  Tier-1-candidate tickers had repairable but messy price rows. They
  were excluded (Task 31 sweep) rather than included with ad-hoc
  patches that would leave a hidden data-quality differential between
  symbols.
- **Tier-2 in the main backtest**: kept only for the ±1-step
  sensitivity band, not as a primary universe.

## Failures encountered in this stage

- The first universe build had a stale docstring vs the actual gate
  values (caught by re-jury-3, Task 25). Fixed; build code and
  docstring re-aligned.
- The first news dedup over-merged ~5.82% of articles (Task 20) —
  see `03_features_and_leakage.md`. Fixed via a stricter dedup key
  and the downstream cascade was re-run.
- A stale "+0.065 IC" disclosure from an earlier version of the
  panel survived into the doc; Task 28 corrected it and logged the
  swing.

## Files of record (this stage)

- `universe/universe_spec.json` — the frozen spec.
- `universe/tier1.txt`, `universe/tier2.txt` — the lists.
- `universe/universe_candidates.csv` — full candidate scoring.
- `universe/news_profile.csv`, `universe/price_profile.csv`.
- `universe/verification_report.md`,
  `universe/verification_pricecheck.csv`.
- `universe/sensitivity_report.md`.
