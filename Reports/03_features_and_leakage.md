# 03 — Causal features and leakage QC

## The panel

`panel/features.parquet` (~870 MB locally; staged to
`/scratch/.../icaif2026/panel/features.parquet` for the HPC
campaign) is the single source of truth for modeling. Rows are
(symbol, date); columns are the 65 input features plus 4 anchor
columns (symbol, date, split, `fwd_ret_H`, `fwd_vol_H`).

Splits (from the dataset inspection):
- **warmup** 2011-01-03 → 2012-12-31 (~437k rows): for 504-row lookbacks.
- **train**  2013-01-02 → 2019-12-31 (~1.53M rows): where per-stock
  z-norm is fit.
- **val**    2020-01-02 → 2020-12-31 (~220k rows).
- **test**   2021-01-04 → 2023-12-28 (~654k rows): the H1 backtest
  window. PREREG §1's `n_eff≈150` non-overlapping H=5 periods is
  reproduced exactly in the live aggregator (`test-split rebalance
  points=149`).

## Causal feature pipeline

`scripts/build_features.py` constructs all 65 features causally:

- **Per-stock z-normalization** is fit ON `train` (2013-2019) ONLY,
  using per-stock running mean/std (the per-stock `feature_norm_stats`
  table is in `panel/feature_norm_stats.parquet`). val and test rows
  are z-scored against the train-fit moments — never their own.
- **Winsorization**: z values clamped to ±8.
- All technical features are computed as functions of *past* prices
  only (no future leak by construction).
- **Forward labels** `fwd_ret_H` and `fwd_vol_H` are constructed for
  every supported H. They are the *targets*, not features, and the
  loader / dataset code never lets them enter the input vector.
- **Sentiment** enters as one optional feature, aligned conservatively
  T+1 from the date-only news Date (PREREG §10).

Outputs: `panel/features.parquet`, `panel/feature_norm_stats.parquet`,
`panel/_feat_log.txt`, `panel/_val_log.txt`.

## The leakage QC battery

`scripts/qc_features_leakage.py` runs the leakage gates as
self-tests. Output: `panel/features_leakage_qc.md`. Categories of
checks (G1 series):

- **G1a (per-symbol monotonicity)** — for each (symbol, feature) the
  value at date d is invariant to the post-d slice of the input
  parquet. Asserted via a re-build from a truncated raw input.
- **G1b (norm-stats fit-window isolation)** — re-computes the per-stock
  z-stats from the train-only slice and checks they match the stats
  stored in `panel/feature_norm_stats.parquet`.
- **G1c (raw-recompute truncation)** — the strongest gate, added in
  Task 30 after re-jury feedback that the earlier version
  effectively re-read the same parquet. The current G1c performs a
  TRUE raw recompute from the FNSPID source (price + news) for a
  random subset of (symbol, date) pairs and asserts the panel rows
  at those (symbol, date)s are bit-identical when the raw is
  truncated at d. This catches any future-peeking caching path that
  the lighter monotonicity check might miss.
- **G1d (target / feature isolation)** — verifies `fwd_ret_H` /
  `fwd_vol_H` never appear under names that the loader's
  `_feature_cols` selector would include.

## Dedup over-merge (Task 20) and its cascade

Re-jury-2 found that the initial article dedup used a loose key
(ticker, normalized title) which collapsed legitimately different
articles posted on different dates (Reuters vs AP rephrasings on
the same headline pattern). Over-merge rate: 5.82%. Effects:
- Mild under-counting of sentiment article counts per (ticker, date).
- A small, biased smoothing of the daily sentiment feature.

Fix (Task 20):
- Stricter key: (ticker, normalized title, **date**, source).
- Re-ran extract → FinBERT scoring → daily aggregation → feature
  build → panel.
- Cascade re-run logged; the panel was rebuilt from scratch.
- Subsequent QC (Task 24) added perturbation coverage to detect any
  similar future regression.

## The stale "+0.065 IC" disclosure (Task 28)

An older version of the panel produced a +0.065 daily-cross-sectional
IC for the sentiment feature; that number ended up in early docs.
After the date-only de-scope and dedup fix, the IC swing was
materially different (smaller / inconsistent across years). Task 28:
- Removed the stale number from the doc.
- Logged the swing transparently in the QC report (per-year and
  pooled).
- Reinforced PREREG §10's "excluded from DSR/PBO/significance"
  treatment so the sentiment feature can never be read as
  inferentially validated.

## What the loader does on top of the panel

`engine/dataset.py` `DateGroupedLoader` is the only access path:

- **Date-grouped batches**: every batch is a block of rebalance dates,
  carrying the full eligible cross-section per date (so the
  cross-sectional Sharpe objective is well-defined per date).
- **H-spaced non-overlapping dates**: rebalance dates subsampled
  every H trading days within `train+val+test` (`closes models-jury F2`).
- **504-row lookback**: per (symbol, date) the input is the 504 most
  recent feature rows ending at date.
- **`sym_id`** (Task 50): a stable per-batch symbol code that the
  net-of-cost turnover loss term needs to align weights across
  consecutive rebalances.
- **`load_panel(split, ...)`** supports `split` as a list (the minimal
  backward-compatible change made in Task 52 so the Phase-2 pool can
  be `train+val+test` in a single loader build).

## Files of record (this stage)

- `panel/features.parquet` (~870 MB local; staged to scratch).
- `panel/feature_norm_stats.parquet`.
- `panel/features_leakage_qc.md`.
- `scripts/build_features.py`,
  `scripts/qc_features_leakage.py`.
- `engine/dataset.py` (loader + CPCV folds).
