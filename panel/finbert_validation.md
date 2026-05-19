# FinBERT Supporting Diagnostic (sentiment is a DE-SCOPED feature)

_2026-05-19. Scope collapsed to C1 (PAPER_PLAN.md): sentiment is NOT a contribution, only an optional on/off feature ablation. This file is a supporting diagnostic, not a claim. GPT-3.5 head-to-head N/A (labels absent in provided data). Stated-date aligned (no tz)._

## 1. Predictive validity (TRAIN-era 2013-2019 only, no leakage)

- pooled Spearman IC: fwd-5d **+0.0101**, fwd-20d +0.0038 (n=495,705 news-day obs).
- daily cross-sectional IC (fwd-5d): mean **+0.0137**, Newey-naive t≈+7.40 over 1757 days.
- decile monotonicity: corr **+0.852**, spread +0.0019.
- Reading: a small but real positive IC means the sentiment feature carries weak return-relevant signal — its incremental ECONOMIC value net of cost/turnover is exactly what the C1 on/off ablation measures (no separate sentiment claim is made; this resolves the prior framing contradiction Jury4 flagged).

## 2. Label audit

- 200-headline stratified gold sample -> `finbert_audit_sample.csv` (human_label blank — manual review pending; advisory only since C3 is dropped).
- finance-lexicon sign agreement (n=60): **0.517** — weak coarse proxy; not load-bearing.

## 3. Dedup: WHY the strict key (Jury5#6)

- This probe measures a property of the RAW DATA: applying the OLD loose key (sym|date|title[:120]) vs a strict (sym|date|fulltitle|url) key to 2,022,342 raw rows over-merges **0.0582**. That is the *motivation*, not the shipped state.
- The shipped corpus is built with the STRICT key (extract_tier1_news.py): 1,295,077 -> 1,826,882 articles (+41% recovered); pipeline over-merge is ~0 by construction. The sentiment panel was rebuilt on it (has_news 24.7% -> ~30%).

## Note
- Title-vs-body ablation (ablate_title_body.py): advisory; title chosen a-priori (51% bodies absent, FNSPID-comparable), not outcome-selected. Decay/H are TRAIN-only (PREREG §9). Sentiment is a feature ablation only.

_elapsed 235s_