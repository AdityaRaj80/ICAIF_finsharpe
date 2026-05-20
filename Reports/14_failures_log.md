# 14 — Cross-cutting failures log

Every failure encountered during the project, with root cause and
resolution. Organised by category, then chronological within
category. The narrative for *why* each failure was found is in
`05_juries_chronological.md`; this file is the consolidated
inventory.

## Data / sentiment

| # | Failure | Root cause | Fix |
|---|---|---|---|
| D1 | UTC→ET point-in-time leakage in daily sentiment (~1 session look-ahead near time-zone boundary) | Joining on date assuming UTC then computing trading-day join in ET | Drop UTC interpretation; align to first trading session STRICTLY AFTER stated date (T+1); full cascade re-run with independent QC. Task 13 / Round-1 jury. |
| D2 | Article dedup over-merged 5.82% of rows (loose key) | Dedup key was (ticker, normalized title) — collapsed different articles same-headline | Stricter key (ticker, normalized title, date, source); cascade re-run. Task 20. |
| D3 | Stale "+0.065 IC" sentiment number in docs after fix | Doc not updated after sentiment de-scope + dedup fix changed the IC | Removed from doc; swing logged in panel QC. Task 28. |
| D4 | Title/body / intraday investigation could be read as feature-search | The exploration produced configurations that an outside reader might think were post-hoc chosen | De-scoped to descriptive-only; PREREG §10 explicitly excludes sentiment Δ from DSR/PBO/significance. Task 24 / Round-3. |
| D5 | Stale-scope mentions in verification_report / ablation / universe_spec | Multiple older docs referenced wider scope before C1 collapse | Doc sweep + corrections. Task 31. |
| D6 | `build_universe.py` docstring vs actual gates inconsistency | Doc said different thresholds than code applied | Doc sweep + correction. Task 25. |

## Engine / models

| # | Failure | Root cause | Fix |
|---|---|---|---|
| E1 | FATAL-1: `softmax(mu/τ)` was NOT the §4 endpoint | At realistic μ scale softmax is near-uniform across cross-section; full-support tilt ≠ top-decile selection | Replaced with differentiable soft-top-decile (sigmoid of `(s - q)/(τ·sd)`, budget-normalised); τ→0 limit equals the hard endpoint (T4 in test_pipeline). Task 46. |
| E2 | FATAL-2: PREREG §9 referenced code that no longer existed (gate, L_meta, vol head, phase schedule, α_pos / β / δ / η) | Earlier head architecture removed; doc not synced | PREREG §9 rewritten to match the *actual* code; CI guard `check_prereg_constants.py` imports code and fails if any value diverges. Task 48. |
| E3 | FATAL-3: loader didn't enforce per-symbol non-overlap of H-day label windows | Date-grouped batching didn't subsample H-spaced dates | `DateGroupedLoader` made H-spaced non-overlapping by construction; `cpcv_folds(purge=1)` ensures train/test disjointness ≥ H trading days. Task 47. |
| E4 | FATAL-4: `verify_models.py` had gameable tests | Tests like "module name contains 'patch'" — an impostor implementation could pass | Re-written with defining-property tests per architecture (ablation degradation, channel-independence permutation invariance, RF computation, etc.). Task 49. |
| E5 | GCformer "global" kernel was effectively local | Decay-pole initialisation produced short-memory poles | Long-memory init `logit(linspace(0.90, 0.9995, K))`; verify_models test asserts pole range. Task 44. |
| E6 | Native ETT anchor pooled-head criterion (iii) failed 7/8 | Pooled encoder + generic linear head is a severe bottleneck for multi-step forecasting — NOT a model bug, a config mismatch | Pre-registered fidelity criterion (iii) PRESERVED un-amended (9a-OUTCOME). Separate native-head 9a-NATIVE re-spec PASS. All-8 derived-gate (k.5) PASS. Tasks 53, 56. |
| E7 | Strengthening pass needed: original capacities were under-tuned | User instruction "don't make models lightweight, make them strong" + jury concern that an honest-negative needs well-built models | Uniform `d_model=256`, more heads/layers per architecture; PREREG §9 re-synced; CI guard re-asserted; verify ALL PASS. Task 51. |

## Objective / endpoint

| # | Failure | Root cause | Fix |
|---|---|---|---|
| O1 | Net-of-cost turnover term missing from training objective | Original `composite_risk_loss` charged no turnover → mismatch with §4-net evaluation | Added L1 turnover term with prev-weight detached; first-rebalance burn-in uncharged; `sym_id` plumbed through loader. Task 50. |
| O2 | First-rebalance turnover charged in eval | Asymmetric vs §4 (deploy-once) | Burn-in uncharged in both `heads.portfolio_returns` (train) and `backtest._net_hard_port` (eval). Task 58 / re-jury-7. |
| O3 | "Uncertainty-aware allocation" claim, but the head was unused in allocation | The risk-aware head trained σ but the allocation only used μ — over-claim | Slimmed to {μ, logvar}; sigma is OPTIONAL NLL calibration ablation; honestly NOT used in allocation. Task 46. |
| O4 | "Surrogate equals endpoint" claim was overstated | Differentiable surrogate optimisation and the hard top-decile rule are different objects | Amendment k.2 dissolves the FATAL: no claim of equality; H1 is precisely the empirical question of whether surrogate moves endpoint; scored exclusively on hard endpoint. Task 55. |
| O5 | H1 endpoint specified in prose but not executable | Code path was ambiguous about soft vs hard endpoint | `engine/backtest.py` `score_h1` made the executable scorer; self-test asserts no `soft_top_decile` reference in call path; hard parity 3e-18 vs canonical endpoint. Task 58. |

## Statistical machinery

| # | Failure | Root cause | Fix |
|---|---|---|---|
| S1 | Overlapping-return inference allowed | H1 endpoint wording permitted H-day overlapping returns → inflated n_eff | §3 / §4 explicitly require strictly non-overlapping H-spaced rebalances; PREREG §1 power statement at n_eff ≈ 150. Task 19, Round-2 fix. |
| S2 | H=5 power statement missing | No quantitative MDE/power analysis pre-registered | Lo 2002 SE; α=0.05, power 0.80; MDE on Sharpe-diff ≈ 0.46/√1; added to PREREG §1. Task 27. |
| S3 | DSR multiple-testing N misspecified earlier | Earlier doc undercounted the searched strategy space | Amendment k.3: `N = 9 × 2 × 64 = 1152`; seeds + CPCV paths excluded as variance reduction. Task 55. |
| S4 | PBO not canonical | Earlier PBO description didn't specify CSCV S | Amendment k.4: CSCV S=10 → C(10,5)=252 train/test recombinations. Task 55. |
| S5 | G1c leakage test was just re-reading the panel | Not a true raw-recompute truncation test | Rewritten as a true raw-recompute from FNSPID source for a random subset of (symbol, date) pairs; truncated raw input. Task 30. |

## Pre-registration

| # | Failure | Root cause | Fix |
|---|---|---|---|
| P1 | "Frozen" framing was overstated mid-development | Doc implied immutability before any training began | Amendment g: "frozen" framing corrected; protocol becomes immutable at first Phase-2 train. |
| P2 | Measured numbers in PREREGISTRATION.md created goalpost-shifting risk | SHA was anchoring numbers, not just a-priori text | Amendment c: all measured numbers removed from doc; doc became pointer-only. Task 29. |
| P3 | Amend-until-pass meta pattern (re-jury-6) | Every failed criterion in a–j was amended → honest-negative non-falsifiable | Amendment **k.1 BINDING-FREEZE CLAUSE**: protocol freezes at first Phase-2 train; failures after this are reported as failures, not amended. No amendment l. Task 55. |
| P4 | Anchor thresholds eyeballed | Numbers in earlier anchor design weren't tied to a rule | Amendment k.5: gates = 1.5 × worst published ETTh1-96 MSE for that architecture class; multiplier fixed BEFORE any run; cited with traceable references. Task 55. |
| P5 | Doc / code constant drift risk | Some §9 constants existed only as plain text in doc | CI guard `engine/check_prereg_constants.py` imports code and fails if any of 35 constants diverge. Task 48. |

## Determinism

| # | Failure | Root cause | Fix |
|---|---|---|---|
| R1 | Toy 2-run determinism check insufficient (re-jury-6 MAJOR) | Real Phase-2 stack (loader + AMP + GradScaler + composite loss + multiple epochs) was unproved | `engine/determinism_real.py` runs the actual Phase-2 path twice on iTransformer + cnn, asserts bit-identical final weights. Task 56. |
| R2 | AMP autocast crashed in `soft_top_decile` (fp16 + torch.quantile) | `torch.quantile` rejects fp16; weight-normalisation unstable in fp16 | `score = score.float()` upcast at top of `soft_top_decile`; safe inside autocast. Task 56. |

## Launcher / HPC

| # | Failure | Root cause | Fix |
|---|---|---|---|
| L1 | SLURM `Invalid job array specification` on `--array=1000-1274` | `MaxArraySize=1001` is a max INDEX limit, not a count | Two chunks per partition: chunk A `--array=0-999`, chunk B `--array=0-274 IDX_OFFSET=1000`; sbatch computes `IDX = $SLURM_ARRAY_TASK_ID + ${IDX_OFFSET:-0}`. Task 52. |
| L2 | First HPO submit accidentally created a partial job grid | The first attempt got 3 "Invalid job array" errors for the B chunks; only chunks A + HPO + AGG queued | `scancel` the partial set; re-push fixed sbatch + submit; resubmit cleanly. Task 52. |
| L3 | Hardcoded `D:\Study\FinSharpe\panel\features.parquet` in `engine/dataset.py` | Path won't resolve on HPC | Made env-overridable: `PANEL = os.environ.get("P2_PANEL", "D:\\...")`. Backward compatible. Task 52. |
| L4 | `optuna` missing in shared `sr_opt` conda env | Env owned by another user; can't modify | `pip install --target=/scratch/.../pylibs "optuna>=3.6"`; `PYTHONPATH=/scratch/.../pylibs:...`. Never $HOME. Task 60. |
| L5 | `engine/dataset.py` `load_panel` filtered on single `split` only | Phase-2 needs the train+val+test pool in one loader build | Made `split` accept list: `flt = ds.field("split").isin(list(split))`. Backward compatible (str still works). Task 52. |
| L6 | Risk: two SLURM mirrors race the same cell | Three partition mirrors (H100/A100/V100) for the same array range | Idempotent skip in `run_cell` / `run_ridge` / `run_hpo`: if output exists, skip and exit 0. Mirrors safe. Task 52. |
| L7 | Risk: HPO straggler blocks 1275 eval tasks | `afterok` would block forever if any HPO task fails | `afterany` dependency; eval cells DEFER (exit 0) if their selected-config JSON missing → operator re-runs `phase2_submit.sh` which (idempotent) only retries the still-missing cells. Task 52. |

## Documentation

| # | Failure | Root cause | Fix |
|---|---|---|---|
| C1 | Multiple doc / code mismatches (verification_report, ablation, spec) | Earlier scope wording not updated after fixes | Sweep + corrections. Tasks 25, 31. |
| C2 | PAPER_PLAN vs PREREG inconsistencies after amendments | Multiple amendments touched both files | Reconciliation pass during re-jury-7. Task 58. |

## What is NOT in this list (deliberately)

- Things that look like failures but are actually disclosed features
  of the dataset / scope (e.g., FNSPID's missing delisted cohort,
  date-only sentiment timestamps, single-dataset external validity).
  These are in `15_limitations_and_threats.md`.
- Anything that would have been an amendment after k.1 — there are
  none by construction; if a post-freeze problem arises it is fixed
  in code (engine bug) and disclosed, but the pre-registered
  criteria are not changed.
