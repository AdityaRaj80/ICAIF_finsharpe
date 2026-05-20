# 02 — News, FinBERT, sentiment pipeline

## Tier-1 news extraction

`scripts/extract_tier1_news.py` parsed the two FNSPID news CSVs,
filtered to the 870 Tier-1 tickers, deduplicated articles, normalized
encodings, and wrote `universe/tier1_news_corpus.parquet`
(~45 MB, several million rows). Log:
`universe/_extract_log.txt`. The corpus columns: ticker, raw_date,
title, body (where available), source, article id.

## FinBERT GPU job on HPC

`scripts/infer_finbert.py` runs ProsusAI/finbert
(`AutoModelForSequenceClassification`) over the corpus producing
3-class softmax probabilities {negative, neutral, positive}. Driven
by `scripts/finbert_job.sbatch` on the BITS HPC.

Critical operational discipline established in this stage and
maintained throughout:

- **Nothing in `$HOME`** — all Hugging Face caches
  (`HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE`), `TORCH_HOME`,
  `XDG_CACHE_HOME`, `TMPDIR` exported to
  `/scratch/goyalpoonam/finsharpe/`. Login-node prep
  (`scripts/hpc_prep.sh`) pre-downloads FinBERT to
  `/scratch/.../models/finbert` so the GPU node runs offline
  (`TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1`).
- **Pylibs on scratch** — `pyarrow` (and later `optuna`) installed
  via `pip install --target=/scratch/.../pylibs` and added to
  `PYTHONPATH`. Never touches `$HOME/.local`.
- The conda env `sr_opt` (`torch 2.11+cu130`) is owned by the
  goyalpoonam account; we only add libs to scratch pylibs, never
  modify the env.

Output: `universe/tier1_news_scored.parquet` (FinBERT
per-article probabilities).

## Daily exp-decay sentiment

`scripts/build_sentiment_daily.py` aggregates the per-article
sentiment to a per-(ticker, date) feature using an **exponential
decay with 5-trading-day half-life**: each article contributes its
(positive − negative) score, weighted by `0.5^(age_in_td / 5)`.
Output: `panel/sentiment_daily.parquet`.

QC: `scripts/qc_sentiment.py` produces
`panel/sentiment_qc_report.md` — distribution per year, per ticker,
NaN audit, gap audit.

## The point-in-time fixup (Task 13)

First QC found the daily aggregation was joining on the news
`Date` field assuming UTC, then computing the trading-day join
against ET trading days. This produced occasional one-session
look-ahead at the time-zone boundary. **Re-jury-1 caught this**.
Fix: drop the UTC interpretation entirely, parse the FNSPID date as
date-only, and align conservatively to **the first trading session
STRICTLY AFTER the stated date (T+1)**. Re-ran the cascade with
an independent QC pass (`scripts/qc_sentiment.py`,
`panel/sentiment_qc_report.md`) — no more boundary leakage.

## Validation against FNSPID GPT-3.5 labels (Task 14)

FNSPID itself ships GPT-3.5-Turbo labels for a subset of articles.
`scripts/validate_finbert.py` cross-tabulated our FinBERT class
predictions against those labels on the overlap subset:
- Confusion matrix per class (in `panel/finbert_validation.md`).
- A 1000-article audit sample
  (`panel/finbert_audit_sample.csv`) where human-readable
  comparisons can be inspected.
The agreement was in the expected literature range for cross-LLM
sentiment labeling on financial headlines (high on negative/positive,
muddier on neutral). Conclusion: FinBERT is fit-for-purpose as a
descriptive feature, not as ground truth.

## Title/body ablation (Task 24)

`scripts/ablate_title_body.py` ran the FinBERT scorer in three
modes: title-only, body-only, title+body. Output:
`panel/title_body_ablation.md`. The differences were small in
aggregate and inconsistent across the year buckets — title-only is
adequate and faster. The ablation result is descriptive only and is
NOT used to justify any modeling decision (otherwise we would have
been smuggling in a post-hoc feature search).

## Re-jury-2 / Task 23: conservative date-only de-scope

Re-jury-2 pressed hard on the date-only timestamp problem. Because
99.5% of FNSPID news rows have no intraday timestamp, the
"publish-minute aware" sentiment story is structurally false. The
fair, honest framing was to **de-scope sentiment** to a single
conservative descriptive feature:

- Sentiment is aligned T+1 strictly-after.
- Sentiment is **one optional on/off feature** in the deep input
  vector; the on/off Δ is reported descriptively only.
- The sentiment-on vs sentiment-off Δ is **explicitly EXCLUDED from
  the DSR/PBO/significance machinery** applied to H1. This is
  formalised in PREREG §10.
- Title/body, intraday, and label-quality investigations are
  reported and then *retired* — they do not feed any inferential
  claim.

## What the sentiment story is NOT in this paper

- Not a tested hypothesis. It is a feature in the input vector for
  both arms.
- Not a publish-minute pipeline. Date-only with T+1 alignment.
- Not used to justify performance differences. The DSR/PBO machinery
  ignores the sentiment on/off Δ entirely (PREREG §10, re-jury-4
  reinforcement).

## Failures and fixes catalogued in this stage

| When | Failure | Fix |
|---|---|---|
| re-jury-1 | UTC→ET boundary leakage in daily sentiment | Date-only conservative T+1 alignment; full cascade re-run; independent QC. |
| Task 20 | Article dedup over-merged 5.82% (loose key) | Stricter dedup key (ticker, normalized title, date, source); re-run; coverage delta logged. |
| Task 24 / re-jury-3 | Title/body intraday investigation could be read as feature-search | De-scoped to descriptive-only; PREREG §10 amended; excluded from DSR/PBO. |
| Task 25 | Stale validate_report wording vs current pipeline | Doc sweep; reports re-generated. |
| Task 31 | Stale-scope mentions in verification_report / ablation / spec | Sweep + corrections. |

## Files of record (this stage)

- `panel/sentiment_daily.parquet`,
  `panel/sentiment_qc_report.md`.
- `panel/finbert_validation.md`,
  `panel/finbert_audit_sample.csv`.
- `panel/title_body_ablation.md`.
- `scripts/finbert_job.sbatch`, `scripts/hpc_prep.sh`,
  `scripts/infer_finbert.py`.
