# 08 — CV, DSR, PBO: the statistical machinery

This is the inference layer. It is what makes H1's accept/reject
robust to (a) leakage across train/test, (b) multiple-testing
inflation across the (model, arm, HPO) search space, and (c)
in-sample/out-of-sample selection bias.

Anchors: López de Prado, *Advances in Financial Machine Learning*
(2018) and *The 10 Reasons Most Machine Learning Funds Fail* (2018);
Bailey & López de Prado, *The Deflated Sharpe Ratio*, 2014; López
de Prado, *Beyond the Sharpe Ratio: Probability of Backtest
Overfitting*, 2016.

## CPCV(6, 2) — the performance estimator

`engine/dataset.py` `cpcv_folds(dates, n_groups=6, k=2, purge=1)`:

- Partition the H-spaced non-overlapping rebalance dates into
  `n_groups=6` contiguous blocks.
- Every k-subset (`k=2`) of those blocks is a TEST path
  → `C(6, 2) = 15` paths total.
- TRAIN dates = everything not in the test blocks AND not within
  `purge=1` rebalance-step (= ≥ H trading days) of any test date.
  This is the **purge+embargo** that kills H-day label overlap
  between train and test.

For each path: train on TRAIN, predict on TEST, collect per-(date,
stock) OOF scores. Concatenating the 15 path-test segments tiles the
full date pool. This is the standard CPCV machinery (López de
Prado's *Combinatorial Purged CV*).

## nested_inner — for HPO early-stop only

`nested_inner(train_dates, n_groups=5, val_groups=1, purge=1)`:
splits a CPCV-fold's TRAIN dates into an INNER train and INNER val.
This is what HPO uses for early-stop; it never touches the outer
test path. Standard nested-CV.

## How HPO uses CPCV

PREREG §9 + k.3 are explicit: HPO selects ONE config per (model,
arm); seeds and CPCV paths are variance reduction of a *fixed
selected config*, not extra selectable strategies. Therefore:

- HPO runs ONCE per (model, arm) on the nested_inner of a FIXED
  designated CPCV path (path 0).
- 64-trial Optuna TPE; median pruning kills hopeless trials early
  (the **sampled** trial count stays 64 — pruning does not change
  the DSR `N`; disclosed).
- Search space: optimizer-only (`lr`, `wd`, dropout, grad-clip,
  patience). Architecture and the pinned objective constants are
  NOT searched. This keeps the controlled-contrast fair across the
  8 backbones.

This is what produces the 17 selected-config JSON files: 8 backbones
× 2 arms = 16, plus Ridge = 17.

## Deflated Sharpe Ratio — `engine/backtest.py` `deflated_sharpe`

Bailey & López de Prado 2014. Given the paired-difference return
series (risk-arm minus comparator), DSR estimates the probability
that the observed Sharpe is meaningfully positive after correcting
for:

- the number of trials searched (`N`),
- the skewness `g3` and excess kurtosis `g4 - 1` of the difference
  series,
- the small-sample finite-`n` adjustment (`sqrt(1/(n-1))`).

`N = (n_models × n_arms) × N_HPO = (9 × 2) × 64 = 1152` is fixed
**by amendment k.3**. The expected-max-Sharpe under N
independent trials uses the standard Gumbel approximation:
`sr0 = sqrt(1/(n-1)) * ((1 - γ) Φ⁻¹(1 - 1/N) + γ Φ⁻¹(1 - 1/(N e)))`,
with `γ = 0.5772156649` Euler-Mascheroni.

Returns `(annualised_diff_Sharpe, DSR_prob)`. `p = 1 - DSR_prob` is
the operational p-value in the k.6 accept rule (`p < 0.05` ≡
`DSR > 0.95`).

## Probability of Backtest Overfitting — `cscv_pbo`

López de Prado 2016, CSCV (Combinatorial Symmetric CV):

- S=10 groups → `C(10, 5) = 252` train/test recombinations
  (PREREG k.4).
- For each recombination, the IS-best configuration is selected on
  the train half; its rank on the test half is logit-transformed;
  PBO is the fraction of recombinations where the IS-best is below
  the OOS median (logit ≤ 0).
- The "configurations" here are the three arms scored
  (risk / mse / ridge), so PBO measures the chance that the H1
  accept came from in-sample-overfitting noise.

PBO is **the primary overfitting control** (k.3): robust to N
mis-specification; DSR with N=1152 is reported as a conservative
secondary.

## ANN factor and the H=5 power statement

`ANN = 252.0 / 5.0` (H=5 → ~50 non-overlapping H-day periods per
year). Per-period Sharpe is annualised by `× sqrt(ANN)`.

Under PREREG §1's power statement (Lo 2002 SE
`SE ≈ sqrt((1 + 0.5 SR²) / n)`; α=0.05, power 0.80, `n_eff ≈ 150`):
the MDE on the Sharpe-difference is ≈ `0.46 / sqrt(1)` in annualized
Sharpe units. `ΔDSR ≥ 0.20` is detectable only if the underlying
annualized Sharpe gap is at least that MDE — disclosed up front.

## The H1 accept rule (k.6)

Operationalised in `engine/backtest.py` `score_h1`:

```python
H1_ACCEPT  iff
    dSharpe_vs_mse   >= 0.20
AND dSharpe_vs_ridge >= 0.20
AND dRankIC_vs_mse   >= 0.01
AND PBO              <= 0.5
AND p_vs_mse         <  0.05         # = DSR_vs_mse   > 0.95
AND p_vs_ridge       <  0.05         # = DSR_vs_ridge > 0.95
```

The two comparators (same-backbone-MSE and Ridge) are both
pre-registered; both must pass. No baseline-selection latitude.

## Files of record (this stage)

- `engine/dataset.py` (cpcv_folds, nested_inner, cscv_folds).
- `engine/backtest.py` (deflated_sharpe, cscv_pbo, score_h1).
- PREREG §1 (power), §9 (CV / HPO / N), k.3 (N=1152), k.4 (PBO),
  k.6 (accept rule).
