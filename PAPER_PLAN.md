# PAPER_PLAN — FNSPID-only relative methodology benchmark (C1)

Decision (user): **FNSPID only**, no second dataset (no CRSP access).
Scope = C1. ICAIF '26 has a **single main track** (8 pp, sigconf, no
supplementary, deadline 2026-08-02); this work is in-scope (the CFP
explicitly lists uncertainty quantification, model validation, trading /
financial forecasting). Contribution type = **empirical-systematization +
an uncertainty-aware-objective increment**, not a new fundamental method.

## The one question (thesis)

> **On the widely-used but contaminated FNSPID dataset, does a
> model-agnostic differentiable cross-sectional portfolio-Sharpe (+epistemic
> uncertainty) training objective change cross-sectional predictive skill
> *relative to* vanilla return-MSE and a Ridge baseline, across eight modern
> architectures, under a leakage-controlled, deflated, PBO-tested
> protocol?**

Contribution: the first rigorous, leakage/deflation/PBO-controlled,
model-agnostic 8-architecture controlled contrast of MSE vs a
Sharpe+uncertainty objective on news-augmented equity data, with FNSPID's
contamination **measured and disclosed**, not ignored (vs ZZR-2020: one
architecture, no deflation; FNSPID: predicted price→R²≈0.98 artifact;
TFT-ASRO: no controls). The carrying contribution is the **honest finding**
(likely: deep objectives do not beat Ridge), not the loss itself.

## Survivorship: disclosed, NOT assumed away (re-jury-4 correction)

We do **not** claim the paired contrast cancels survivorship. Two
objectives select different sub-portfolios with different exposure to the
missing-delisted mass → the paired difference carries a **residual,
possibly self-favorable, survivorship-interaction bias**. We (i) state the
mechanism/direction as a primary limitation, (ii) sign-bound it with the
thin internal early-stop/delisted cohort, (iii) keep all absolute/level
metrics out of every claim. The contribution is the controlled *relative*
measurement with this bias disclosed — not a bias-free economic result.

## Primary endpoint (single, falsifiable, one horizon, one comparator)

H1: for a given backbone, the portfolio-Sharpe(+uncertainty) objective
improves the **paired** test cross-sectional **rank-IC** and the
**Deflated Sharpe of the return-difference series** vs (a) the same
backbone with MSE and (b) **Ridge on the identical features** — the
single pre-registered comparator (other §-baselines are descriptive
context only, never the H1 reference: no baseline-selection leakage).
ΔDSR ≥ 0.20, Δrank-IC ≥ 0.01, p<0.05 post-PBO, **H = 5** (n_eff≈150;
PREREG §3). Allowed to fail; a null is the informative answer.

## Universe (fixed, audited, disclosed)

The **fixed audited 870-name Tier-1 list IS the backtest universe**. It
was full-sample-selected (completeness/liquidity/news gates) → its
selection survivorship/look-ahead is a **disclosed limitation**, only
partially mitigated by the paired endpoint, not eliminated. All features /
per-stock z-norm / leakage-QC are on exactly this 870 panel (no
spec-vs-artifact gap). ±1-economic-step band reported.

## Models & baselines

8 backbones {iTransformer, PatchTST, TFT, GCFormer, DLinear, LSTM, RNN,
CNN} × {MSE, portfolio-Sharpe+uncertainty}, one identical harness.
**Ridge (identical features) = the single H1 comparator.** Descriptive
context baselines: x-sec momentum, buy&hold, ZZR-2020 direct-Sharpe,
ex-post vol-targeting (NOT H1 references).

## Absolute economic metrics → appendix only

Dollar/level Sharpe, equity curves: appendix, labeled "within a disclosed
survivor universe — levels NOT investable; completeness only." No headline
rides on an absolute number.

## Robustness arms (NOT headline; descriptive-only)

- C2 de-risking overlay vs ex-post vol-targeting at equal turnover.
- Sentiment feature on/off — FNSPID-only, date-only, conservatively
  aligned; weak train IC (~+0.010 pooled / ~+0.014 daily-XS); **the on/off
  Δ is descriptive-only, EXCLUDED from DSR/PBO/significance** (its
  label/intraday/title-body validation was deliberately dropped, so the
  relative framing must not be read as validating it).
- Thin internal delisted/early-stop cohort — small signed *bound*, not a
  cure. ±1-step universe band; 17 repairable-row tickers excluded.

## Limitations (up front, in the abstract)

FNSPID: no clean delisted cohort (2020 snapshot artifact) → absolute
performance non-investable; **residual survivorship-interaction bias in
the paired endpoint, disclosed not cancelled**; 2021 news trough; 99.5%
date-only timestamps (sentiment a weak de-scoped feature); single dataset
→ limited external validity. Framed as a methodology re-analysis, not an
absolute-performance claim.

## Positioning

Anchors: Moody & Saffell 1998; Zhang–Zohren–Roberts 2020; López de Prado
(purged CV / DSR / PBO). FNSPID = the contaminated public benchmark we
re-analyze rigorously. TFT-ASRO: not in positioning.
