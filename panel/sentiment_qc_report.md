# Sentiment Panel QC (conservative date-only rule)

_2026-05-19. Sentiment is a DE-SCOPED optional feature (PREREG §10), not a contribution. Rule: effective session = first trading day STRICTLY AFTER the stated date (no intraday assumption)._

**VERDICT: PASS** — 2,844,044 rows, 870 symbols.

## CRITICAL DISCLOSURE — FNSPID timestamp resolution

- **99.53%** of scored `ts` are `00:00:00 UTC` (date-only). FNSPID `Date` has **no usable intraday resolution**; any intraday/point-in-time precision claim is unsupported. The conservative strict-next-session rule is therefore the only honest alignment. Per-year date-only fraction:
| year | date-only |
|--|--|
| 2009 | 100.0% |
| 2010 | 100.0% |
| 2011 | 100.0% |
| 2012 | 100.0% |
| 2013 | 100.0% |
| 2014 | 100.0% |
| 2015 | 100.0% |
| 2016 | 100.0% |
| 2017 | 100.0% |
| 2018 | 100.0% |
| 2019 | 99.9% |
| 2020 | 94.3% |
| 2021 | 100.0% |
| 2022 | 100.0% |
| 2023 | 99.6% |

## Gates

- PASS — G1 EXACTLY 0 look-ahead (effective session strictly AFTER stated date)
- PASS — G2 every year has news (>0); 2021 trough disclosed
- PASS — G3a news-day days_since_news==0 (exact)
- PASS — G3b news-day sent_decay==sent_raw (exact)
- PASS — G3c no-news decay monotone non-increasing (exact)
- PASS — G4 sent_decay in [-1,1], variance>0
- PASS — G5 macro 2020-03 COVID dip vs Dec19-Feb20
- PASS — G6 independent re-derivation == builder (0 mismatched syms)
- PASS — G7 ts date-only fraction MEASURED & disclosed (not gated, reported)

## Point-in-time (150-symbol independent re-derivation)
- look-ahead violations (effective <= stated date): **0** (must be 0).
- builder vs independent session-set mismatch: **0** (must be 0).

## Per-year coverage

| year | rows | %news | mean intensity |
|--|--|--|--|
| 2011 | 219,240 | 16.49 | 0.46 |
| 2012 | 217,505 | 21.07 | 0.42 |
| 2013 | 219,240 | 22.35 | 0.45 |
| 2014 | 219,241 | 24.45 | 0.52 |
| 2015 | 219,240 | 32.93 | 0.72 |
| 2016 | 219,240 | 36.01 | 0.76 |
| 2017 | 218,372 | 35.14 | 0.79 |
| 2018 | 218,376 | 42.32 | 1.02 |
| 2019 | 219,240 | 41.37 | 0.96 |
| 2020 | 220,110 | 28.5 | 0.6 |
| 2021 | 219,240 | 21.2 | 0.36 |
| 2022 | 218,370 | 32.11 | 0.58 |
| 2023 | 216,630 | 36.09 | 0.72 |

## Decay / distribution / macro
- dsn==0 True; decay==raw True; monotone True; sent_decay mean 0.0594 std 0.3170; COVID 2020-03 -0.1252 vs base 0.1031.

## Note
- Prior UTC->ET 'pre-open' treatment WITHDRAWN (theatre on a date-only field; was mildly anti-conservative). Sentiment used only as an on/off feature ablation.