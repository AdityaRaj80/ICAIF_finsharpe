# 05 — Adversarial juries (chronological)

The project's quality control was driven by repeated harsh
adversarial-jury rounds. Each jury was instructed to find FATAL,
MAJOR, and MINOR problems with the work as it stood at that moment.
Most amendments in `04_preregistration_amendments.md` were responses
to one of these juries. This file is the chronological inventory.

## Round 1 — initial full-scale ICAIF jury (Task 11)

Snapshot: dataset, universe, sentiment pipeline up; modelling
pipeline not yet written.

Key findings:
- **FATAL** — UTC→ET point-in-time leakage at the time-zone boundary
  in daily sentiment (~1 session look-ahead on edge cases).
- **MAJOR** — universe selection criteria not yet documented in
  sufficient detail; sensitivity band not yet established.
- **MAJOR** — sentiment QC was the same person who built the
  pipeline; an independent QC was missing.

Responses (Tasks 12-14):
- Sentiment switched to date-only conservative T+1; full cascade
  re-run; **independent** QC pass added
  (`scripts/qc_sentiment.py`).
- Universe sensitivity report added.
- FinBERT vs GPT-3.5 cross-validation added.

## Round 2 — fixed-pipeline jury (Task 21)

Snapshot: post-Round-1 fixes.

Key findings:
- **FATAL** — article dedup was over-merging 5.82% of rows.
- **MAJOR** — scope (C1 + C2 + economic claim) too broad for 8 pages
  and the disclosed survivor universe.
- **MAJOR** — H1 endpoint wording allowed H-overlapping return
  windows, inflating n_eff.

Responses (Tasks 20, 22):
- Stricter dedup key; cascade re-run.
- Scope **collapsed to C1**; PAPER_PLAN + PREREG rewritten.
- §3 / §4 explicitly require strictly NON-overlapping H-spaced
  rebalances → honest n_eff ≈ 150 at H=5.

## Round 3 — collapsed-scope jury (Task 26)

Snapshot: post-Round-2; C1 only; protocol pinning H=5.

Key findings:
- **MAJOR** — H=5 primary power statement missing.
- **MAJOR** — stale "+0.065 IC" sentiment number in the doc no
  longer matched the post-fix pipeline.
- **MAJOR** — measured numbers in PREREGISTRATION.md created
  goalpost-shifting risk (the SHA was anchoring numbers, not the
  a-priori text).
- **MINOR** — G1c leakage test was just re-reading the panel, not a
  true truncation test.
- **MINOR** — stale-scope mentions in verification_report /
  ablation / spec.

Responses (Tasks 27-31):
- Pre-registered MDE/power statement added at H=5.
- Stale +0.065 IC removed; swing logged.
- All measured numbers stripped from the protocol doc; doc became
  pointer-only.
- G1c rewritten as a true raw-recompute truncation test.
- Doc sweep fixed stale-scope mentions.

## Round 4 — FNSPID-only reframe jury (Task 34)

Snapshot: after scope collapse + the FNSPID-only decision (no CRSP
access; second-dataset path closed).

Key findings:
- **MAJOR** — the prior framing implicitly claimed that the paired
  contrast cancels survivorship. It does not (different objectives
  select different sub-portfolios with different exposure to the
  missing delisted mass → residual self-favoring bias).
- **MAJOR** — absolute economic metrics in the main story still
  implied investability.
- **MAJOR** — sentiment de-scope wording was not strong enough.

Responses (Task 35, the 5 bounded correctness fixes):
- Survivorship: residual interaction-bias disclosed as a primary
  limitation in PAPER_PLAN; sign-bounded by the thin internal
  early-stop/delisted cohort; all absolute/level metrics kept out of
  every claim.
- Absolute metrics moved to appendix with "levels NOT investable;
  completeness only" labels.
- Sentiment on/off Δ explicitly EXCLUDED from DSR/PBO/significance.
- Re-hash of PREREGISTRATION.md.

## Round 5 — pre-training models jury (Task 39, then Tasks 45)

Snapshot: 8 in-house backbone implementations done; not yet trained.

Key findings:
- **FATAL-1 (the big one)** — `softmax(mu/τ)` head was NOT the §4
  endpoint. At realistic μ scale the softmax was near-uniform; the
  "soft top-decile / image of §4" claim was mathematically false.
- **FATAL-2** — PREREG §9 referenced code paths that no longer
  existed (gate, L_meta, vol head, phase schedule).
- **FATAL-3** — the loader did not enforce per-symbol non-overlap of
  the H-day label windows.
- **FATAL-4** — `verify_models.py` had gameable tests that the wrong
  implementation could pass.

Responses (Tasks 40, 44, 46-49):
- **FATAL-1**: replaced with the differentiable
  **soft-top-decile** operator whose τ→0 limit provably equals the
  §4 hard long-only top-decile equal-weight portfolio. See
  `07_objective_and_endpoint.md`.
- **FATAL-2**: PREREG §9 rewritten to match the *actual* code; CI
  guard (`engine/check_prereg_constants.py`) imports the code and
  FAILS if any pinned value diverges; §4 contradiction fixed.
- **FATAL-3**: `engine/dataset.py`
  `DateGroupedLoader` enforces non-overlapping H-spaced rebalance
  dates; CPCV fold generator (`cpcv_folds`) added with purge.
- **FATAL-4**: `engine/verify_models.py` re-written with decisive
  non-gameable tests (defining-property checks per architecture).
- 8 models re-checked individually for defining-property fidelity.

## Round 6 — corrected-engine jury (Tasks 51, 54, 55)

Snapshot: post-FATAL fixes; models strengthened to research-grade
capacity (d_model=256, more layers, etc., per the user instruction
"don't make the models lightweight, make them strong").

Key findings:
- **MAJOR** — net-of-cost turnover term missing from the training
  objective (the trained objective did not match §4-net; misalignment
  between train and eval).
- **MAJOR** — external ETTh1 sanity anchor missing; honest-negative
  not credible without architecture fidelity evidence.
- **META** — the amendment chain a–j showed an
  "amend-until-pass" pattern; this is the source of un-priced
  structural risk for an honest-negative claim.

Responses (Tasks 50, 53, 55, 56):
- Net-of-L1-turnover term added to `composite_risk_loss`;
  `sym_id` plumbed through the loader; first-rebalance burn-in
  uncharged (matches §4 deploy-once).
- ETTh1 anchor designed pre-result (criteria i–iv fixed), ran;
  criterion (iii) FAILED 7/8 (preserved); native-head re-spec PASS;
  all-8 mechanically-derived gates PASS.
- **Amendment k FINAL/BINDING** introduced (see
  `04_preregistration_amendments.md`):
  k.1 binding-freeze, k.2 surrogate dissolution, k.3 DSR-N=1152,
  k.4 CSCV PBO, k.5 derived anchor thresholds, k.6 pre-registered
  primary decision rule.

## Round 7 — final binding-engine jury (Task 57)

Snapshot: post-amendment-k; engine fully fixed; CPCV/DSR/PBO scorer
designed; real-path determinism proven.

Key findings (re-jury-7, the last one before launch):
- **FATAL** — the H1 endpoint scorer was specified in PAPER_PLAN
  prose but NOT EXECUTABLE; the existing `score_h1` call path was
  ambiguous about whether it routed through the soft surrogate or
  the hard endpoint.
- **MAJOR** — `tau_schedule` wording in PREREG implied k.2 dependence
  on annealing; that dependence does not exist (validity is on the
  hard endpoint; static τ=0.05 is fine).
- **MAJOR** — k.5 cited literature numbers without traceable
  citations.
- **MAJOR** — protocol doc and PAPER_PLAN had small consistency
  contradictions left from earlier passes.
- **MAJOR** — first-rebalance turnover was being charged in the
  backtest; that double-charges the initial deployment vs §4
  (deploy-once).
- **MAJOR** — real-path determinism (Phase-2 loader+AMP+GradScaler+
  composite loss, not toy) was unproved.

Responses (Task 58):
- `engine/backtest.py` made the executable §4 scorer. `score_h1`
  routes through `hard_top_decile_returns` unambiguously; the
  self-test asserts no `soft_top_decile` reference is in the call
  path; parity vs the canonical endpoint asserted at 3e-18.
- PREREG wording clarified: validity does NOT depend on `tau_schedule`.
- k.5 citations added with traceable references.
- Doc reconciliation pass.
- First-rebalance burn-in: uncharged (PREREG k.2 / the turnover
  fix); same in train and eval.
- `engine/determinism_real.py` runs the actual Phase-2 path twice
  on iTransformer + cnn; final weights asserted bit-identical;
  report at `bench/determinism_real_report.md`.

## Round 8 — would-be-next jury (DEFERRED)

By the time re-jury-7 closed, the amendment chain was at k.1
binding-freeze. The next adversarial gate is the **post-result
jury** that the autonomous monitoring loop will spawn when the H=5
aggregator finishes (`13_freeze_and_campaign.md`). That jury is on
the H1 verdict's soundness (alignment with k.6, DSR/PBO sanity,
leakage residual, internal consistency); it does NOT amend the
protocol (k.1 forbids that). If it flags a hard-blocking error, we
fix the bug and re-run; if it flags an interpretive concern, we
disclose it in the paper alongside the verdict.

## What every jury was told

The standing jury instruction:
- Be harsh; find FATAL, MAJOR, MINOR problems.
- Treat any silent disagreement between docs and code as a problem.
- Treat any unmotivated parameter as a problem.
- Treat any criterion that quietly slipped between amendments as a
  problem.
- A null result is a legitimate outcome; any pattern that converts
  a null into a positive is a problem.
