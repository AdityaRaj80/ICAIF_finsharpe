# PHASE-2 ANALYSIS PROTOCOL — FinSharpe / ICAIF 2026

**Status & honest framing (re-jury-4 fix).** This is NOT yet "frozen": it
was iterated several times pre-execution as adversarial review surfaced
problems (history fully disclosed, append-only, §11 a→e). It contains
**no measured results** — every QC number lives in dated `panel/` &
`universe/` reports. The protocol **becomes immutable at one moment: the
instant the first Phase-2 model is trained**; the SHA256 in
`PREREGISTRATION.sha256` is re-stamped then and the doc is not edited
afterward. Pre-execution iteration is legitimate and disclosed, not hidden;
post-execution editing would be misconduct. No `.git`; "a-priori" labels
are author-asserted, anchored by the final hash + the §11 log.

**Held out by this freeze:** ALL Phase-2 model training, HPO, the
portfolio backtest, every H1 test. None of this has been run.

**Scope: C1 only.** One pre-registered question (PAPER_PLAN.md). C2 =
robustness arm; C3 dropped; sentiment = one optional date-only feature.

**Dataset (amendment d — FNSPID-only).** Single dataset = **FNSPID**, a
widely-used but contaminated public benchmark (no clean delisted cohort /
2020 snapshot artifact; 2021 news trough; 99.5% date-only timestamps), used
under full disclosure. No second/clean dataset (no CRSP access). The paper
is a leakage-controlled **relative methodology re-analysis**: the primary
endpoint is a **paired contrast on a single fixed universe**. Survivorship
is **NOT claimed to cancel** (re-jury-4 correction): two different
objectives select different sub-portfolios with different exposure to the
missing-delisted mass, so the paired difference carries a **residual,
possibly self-favorable, survivorship-interaction bias**. We do not assert
neutrality; we (i) state the bias direction and mechanism as a primary
limitation, (ii) bound it with the thin internal early-stop/delisted
cohort as a signed sensitivity, (iii) keep absolute/level performance out
of all claims (appendix, non-investable). The contribution is the
leakage/deflation/PBO-controlled *relative* measurement with this bias
disclosed, not a bias-free economic result.

---

## 1. Single primary hypothesis (sharp, falsifiable)

**H1.** For a given backbone, the differentiable cross-sectional
**portfolio-Sharpe objective** improves the **paired** test cross-sectional
**rank-IC** and the **Deflated Sharpe of the return-difference series**
vs (a) the same backbone with vanilla return-MSE and (b) **Ridge regression
on the identical feature set** — the single, pre-registered comparator
(not a post-hoc "best" baseline; the other baselines in §6 are descriptive
context only and are NOT the H1 reference, removing max-statistic /
baseline-selection leakage). Thresholds: **ΔDSR ≥ 0.20** and
**Δrank-IC ≥ 0.01** at **p < 0.05 post-PBO**, **at H = 5** (§3 power),
long-only, 10 bps cost, on FNSPID. The endpoint is the paired *difference*
(never an absolute level); its residual survivorship-interaction bias is
disclosed and bounded per the header, not assumed away.

Refutation: not met at H=5 → H1 **rejected, reported as a null**. This is
one question with one binding horizon; a null is an informative answer to
*that* question, not a fallback for other claims.

## 2. Universe (per dataset; absolute a-priori floors)

Absolute economic / data-hygiene floors, NOT outcome-tuned: median ADV ≥
$5M; (FNSPID arm only) total news ≥ 500; price history covers warmup+test;
n_rows hygiene; max gap ≤ 10 d; curated ETF exclusion; common shares only.
The **fixed audited 870-name Tier-1 list IS the backtest universe**
(re-jury-4 fix: the prior "point-in-time reconstruction / 870 = superset
only" wording is withdrawn — it created a spec-vs-artifact gap where every
leakage-QC certified a universe the paper would not use, with no real
survivorship gain since FNSPID has no delisted names anyway). The 870 was
selected with full-sample completeness/liquidity/news gates; this
**universe-selection survivorship/look-ahead is a disclosed limitation**,
only partially mitigated by the paired endpoint, NOT eliminated. All
features, per-stock z-norm stats, and leakage-QC are on exactly this 870
panel. Robustness reported on a ±1-economic-step band; empirical percentile
each floor sits at is in the universe QC report (post-hoc).

## 3. Splits, CV, and statistical power (re-jury-3 FATAL fix)

- Window: warmup 2y / inner pool 2013–2020 / held-out test 2021–2023.
  **No separate calendar-2020 validation block.** Model selection / HPO /
  early-stop use ONLY nested Combinatorial Purged CV inner folds within
  the inner pool (rank-IC on inner sub-folds). Purge + embargo = H td both
  sides of every boundary.
- **Power (binding).** PRIMARY backtest = strictly NON-OVERLAPPING
  single-schedule rebalancing every H td → n_eff = ⌊test_td/H⌋ independent
  H-period returns. For H=5 over ~750 test td, n_eff ≈ 150 — adequate for
  DSR and a block-PBO. **H1 inference is restricted to H = 5.**
  - **Pre-registered minimum detectable effect.** At n≈150, two-sided
    α=0.05, power 0.80, the MDE on the Sharpe-difference is ≈ 0.46/√1 in
    annualized-Sharpe units (Lo 2002 SE ≈ √((1+½SR²)/n)); ΔDSR≥0.20 is
    detectable only if the underlying annualized Sharpe gap ≳ that MDE —
    stated up front so a null is interpreted as "no effect ≳ MDE," not
    "proven zero."
  - H ∈ {20,63,126,252}: **descriptive only** — point estimates +
    stationary block-bootstrap CIs from the daily-rebalanced overlapping
    portfolio with Newey–West HAC (lag = H−1). **No DSR/PBO/DM/MCS, not in
    H1, not a significance claim** (CSCV is combinatorially infeasible at
    n_eff≈37 for H=20 and below). n_eff = ⌊test_td/H⌋ reported per cell.
- Thesis wording is bound to H=5; "any/multi-horizon" significance
  language is banned (multi-horizon appears only as descriptive context).

## 4. Portfolio backtest

Daily cross-sectional panel. Signal → within-date rank → **long-only
top-decile equal-weight (primary)**; decile long-short = appendix (borrow
cost). Rebalancing = non-overlapping single schedule (§3). Costs: grid
{0,5,10,20} bps + square-root market-impact at stated AUM. Metrics:
Sharpe (NW + block-bootstrap CI), Sortino, MaxDD, Calmar, CVaR₅, turnover.
Primary stat = Deflated Sharpe of the **paired return-difference series**,
deflated by the full configuration count (§9). Primary endpoint is the
**relative contrast** (paired Δrank-IC, DSR-of-difference) — survivorship
**common-mode** (enters both arms identically, first-order cancels in the
difference). All absolute/level Sharpe → appendix, labeled
non-investable; the survivorship limitation is disclosed in the abstract.

## 5. Models & baselines

8 backbones {iTransformer, PatchTST, TFT, GCFormer, DLinear, LSTM, RNN,
CNN} × {MSE, portfolio-Sharpe objective}, one identical harness. Baselines
every fold: Ridge (same features), x-sec momentum, buy&hold,
Zhang–Zohren–Roberts-2020 direct-Sharpe, ex-post vol-targeting.

## 6. HPO parity

Identical for every model AND Ridge: N_HPO = 32 TPE trials, early-stop =
nested inner-CPCV rank-IC (never test, never a calendar block).
Convergence curves published; if any deep model is still improving at 32
trials that is disclosed (so "deep loses" is not an under-tuning artifact).

## 7. Determinism & seeds

Seeds {0,1,2,3,4}; deterministic algos; cuDNN deterministic; seeded
loaders. Every headline number = mean ± std over 5 seeds; within ±1 std =
"not separable".

## 8. Feature normalization (a-priori)

Per-stock z-score fit on the **2013–2019 sub-window only** (intentionally
tighter than the 2013–2020 inner pool; never sees test), applied to all
splits; winsorize z to ±8. (Clarifies the code/spec boundary the ML juror
flagged.)

## 9. Frozen constants & multiple-comparison

Constants author-set a-priori, fixed (sentiment carry: conservative T+1 on
the STATED date — no tz, FNSPID Date is date-only; SEQ_LEN 504; loss
coeffs β.5 δ.3 η.1 α_pos 10; phase schedule; gate temp; exp-decay
half-life 5). Any train-only search is nested-CV only and logged. Full
reported configuration count = 13 models × 2 objectives × 5 seeds (H=5
primary) + descriptive/robustness arms; **DSR deflation N = full
cardinality**; one primary table; all else appendix with
Benjamini–Hochberg FDR.

## 10. Sentiment feature (de-scoped)

FNSPID `Date` is date-only; sentiment is aligned conservatively (first
session STRICTLY AFTER the stated date) and enters only as one on/off
feature ablation, never a claim. **The sentiment-on vs sentiment-off Δ is
reported DESCRIPTIVELY ONLY and is explicitly EXCLUDED from the
DSR/PBO/significance machinery applied to H1** (re-jury-4 fix: although it
is also a paired difference, it does NOT receive the inferential treatment
of the primary contrast, so the relative framing cannot be read as
silently validating sentiment, whose label/intraday/title-body validation
was deliberately dropped). Its weak signal and the date-only limitation
are in the (post-hoc) sentiment QC. Prior UTC→ET treatment withdrawn.

## 11. Amendment log (append-only)

- 2026-05-19 a: initial freeze (3-leg).
- 2026-05-19 b: re-jury-2 → scope collapsed to C1; renamed; §3/§4
  overlapping-return inference fixed; §3/§7 early-stop reconciled;
  sentiment → conservative date-only.
- 2026-05-19 c: re-jury-3 → (i) all measured numbers removed from this
  doc (now pointer-only; SHA anchors a-priori text only); (ii) H1 binding
  horizon restricted to **H=5** with a pre-registered MDE/power statement
  (n_eff≈37 at H=20 conceded underpowered, demoted to descriptive);
  (iii) two-dataset design added — clean PIT/delisting-inclusive dataset
  is PRIMARY, FNSPID is the contaminated-replication arm; (iv) disclosed:
  the sentiment alignment fix moved the FNSPID train IC from a prior
  anti-conservative **+0.065** to the conservative **+0.010**; the ~6.5×
  delta was removed look-ahead, not signal — recorded here, not buried.
- 2026-05-19 d: user decision → **FNSPID-only** (no CRSP access). The
  amendment-c two-dataset design is WITHDRAWN. Reframed as a
  leakage-controlled relative methodology re-analysis of the contaminated
  public FNSPID benchmark: primary endpoint = the **paired
  survivorship-common-mode contrast** (Δrank-IC, DSR-of-difference,
  H=5); operational universe reconstructed point-in-time at each rebalance
  (870 list = descriptive superset only); all absolute/level metrics
  demoted to a non-investable appendix; single-dataset external-validity
  limitation disclosed in the abstract. No new measured numbers introduced.
- 2026-05-19 e: re-jury-4 bounded fixes. (i) "survivorship common-mode /
  first-order cancels" **RETRACTED** — replaced by an explicit, disclosed,
  possibly self-favorable residual-interaction bias (mechanism stated,
  signed-bounded by the thin internal delisted cohort). (ii) H1 comparator
  pinned to **Ridge on identical features** (single pre-registered
  reference; kills baseline-selection leakage). (iii) PIT-universe
  reconstruction **withdrawn**: the audited fixed 870 IS the backtest
  universe; its full-sample-selection survivorship/look-ahead is a
  disclosed limitation (closes the spec-vs-audited-artifact gap).
  (iv) sentiment on/off Δ is descriptive-only, **excluded from
  DSR/PBO/significance**. (v) "frozen" framing corrected → immutable only
  at first Phase-2 train; pre-execution iteration disclosed not hidden.
  No new measured numbers introduced.
