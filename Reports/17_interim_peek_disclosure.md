# 17 — Interim-peek disclosure (research-integrity)

This file exists because, during the multi-day Phase-2 campaign, the
author looked at PARTIAL results before the full pre-registered run
completed. Pre-registration credibility requires that such peeking be
disclosed, not hidden. An adversarial jury (the post-launch jury,
`05_juries_chronological.md` "Round 8") flagged this explicitly; this
document is the response.

## What happened

- While the H=5 campaign was running, a LITE partial tool
  (`engine/phase2_quicklook.py`) was used to compute per-(backbone,arm)
  annualised hard-top-decile Sharpe and rank-IC on whatever seed-0 eval
  scores existed so far — for iTransformer and DLinear.
- Those interim numbers were reported to the operator and committed to
  the public git repo (`bench/quicklook.md`, carrying a bold
  "NOT the H1 verdict / PARTIAL" header).
- The operator, on seeing small numbers, asked once whether the Sharpe
  ratio could be increased. The response was an explicit REFUSAL to
  tune anything post-freeze, citing PREREG k.1 (the binding-freeze
  clause); see `04_preregistration_amendments.md`.

## Why this is bounded (what the peeking did NOT do)

The PREREG k.1 freeze protects the **criteria** and the **execution
grid**, and neither was touched:

- The k.6 accept rule, the DSR/PBO machinery, the thresholds, N=1152,
  the H=5 horizon — all SHA-frozen in `PREREGISTRATION.md`; no interim
  number can change them.
- The execution grid is fully enumerated: 8 backbones × 2 arms × 5
  seeds × 15 CPCV folds + Ridge = 1275 cells, ALL of which run. No
  interim number changed which cells run, which seeds, which folds, or
  the order in a way that affects results (idempotent skip + a fixed
  dispatch map).
- No hyperparameter, architecture, feature, universe, objective
  coefficient, or τ was changed. HPO configs are selected by the frozen
  64-trial TPE; the operator's "raise Sharpe" question was declined.

## The residual risk we DO disclose

What peeking cannot be fully insulated from is **narrative anchoring** —
having seen "DLinear paired ΔSharpe +0.509," the author and operator are
now aware of a favourable-looking partial. This can bias how the final
result (especially a null) is *written up*. We disclose this rather than
pretend it away. Mitigations:

1. The interim quicklook numbers are **non-evidential** by construction
   — `phase2_quicklook.py` computes no DSR, no PBO, no Ridge comparator,
   no k.6 rule. The H1 verdict is produced ONLY by
   `engine/phase2_aggregate.py` → `backtest.score_h1` on the complete
   1275-cell grid.
- 2. Every interim peek is logged here and in the session transcript.
3. The paper's "Protocol iteration history" section (re-jury-7
   mandate) will additionally state: interim directional peeks occurred;
   the operator asked once about raising Sharpe; the request was
   declined under k.1; no interim number altered any frozen criterion
   or any executed cell.

## Corrections the peek surfaced (kept on the record)

The interim analysis also produced two author errors, corrected and
preserved here rather than silently fixed:

- **Portfolio size misstated.** The §4 long-only top-decile holds
  ~86 names (10% of the ~852-name eligible cross-section), NOT the
  "~8-9 names" stated in one interim explanation. Corrected.
- **"dlinear/mse is actively harmful" — overstated.** Its −0.31
  absolute test-split Sharpe is dominated by long-only exposure to the
  2021–2023 universe (incl. the 2022 bear market), not by model
  failure. The model's genuine cross-sectional ranking deficit is the
  small rank-IC (≈ −0.028). Absolute long-only Sharpe is market-drift
  dominated — which is exactly why H1 is a PAIRED difference and why
  PAPER_PLAN keeps absolute levels appendix-only.

## Speculative observations explicitly NOT claimed

A post-hoc "the training objective flips the sign of cross-sectional
skill, most for the simplest model" narrative was floated internally on
2 backbones at 1 seed. It is **not pre-registered, not tested, N=2**,
and is recorded here ONLY as a hypothesis-generating observation. It is
excluded from any results claim unless it survives the full grid +
DSR/PBO; if it does not, it is not mentioned as a finding.

## Status

The interim handling is disclosed. The eventual H1 verdict rests on the
frozen criteria + the full grid + the self-tested scorer, none of which
the peeking touched. This document is part of the honest record.
