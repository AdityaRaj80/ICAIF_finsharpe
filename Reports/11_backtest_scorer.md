# 11 — The H1 / backtest scorer (`engine/backtest.py`)

This is the file that produces the H1 verdict. Re-jury-7 (the final
pre-launch round) ruled the earlier H1 endpoint to be specified in
prose but **not executable**; making it executable + self-tested was
the resolution (Task 58).

## What the scorer is contractually

Given a trained model's per-(date, stock) test scores, the scorer:

1. Forms the §4 portfolio with **`hard_top_decile_returns`** —
   long-only, equal-weight top decile — NEVER the soft surrogate.
2. Subtracts one-way turnover cost on the L1 weight change vs the
   previous rebalance; the FIRST rebalance is **UNCHARGED**
   (burn-in / deploy-once, PREREG k.2 / re-jury-7 turnover fix).
3. Computes the per-arm annualised Sharpe and the **paired
   difference series** (risk minus mse, risk minus ridge).
4. Computes the **Deflated Sharpe** of each paired difference series
   (Bailey & López de Prado 2014) deflated by `N = 1152`
   (PREREG k.3).
5. Computes **PBO via CSCV** with `S=10` → 252 train/test
   recombinations (PREREG k.4).
6. Computes per-arm rank-IC and Δrank-IC.
7. Returns booleans for the **pre-registered k.6 accept rule**.

## Constants

```python
from heads import hard_top_decile_returns, FRAC, COST
from dataset import cscv_folds

ANN   = 252.0 / 5.0          # H=5 -> ~50 non-overlapping periods/yr
DSR_N = 1152                 # PREREG §12 k.3
EUL   = 0.5772156649         # Euler-Mascheroni
```

## `_net_hard_port(score, y, did, sym_id, cost=COST)`

The per-period return of the HARD top-decile endpoint, net of
one-way turnover with the FIRST rebalance uncharged. Long-only
equal-weight top decile by model score.

```python
for d in unique(did):
    s, ys, ss = score[mask], y[mask], sym[mask]
    kk  = max(1, ceil(FRAC * len(s)))
    top = argsort(-s)[:kk]
    w   = zeros(n_syms); w[idx(ss[top])] = 1.0 / kk
    gross = ys[top].mean()
    out.append(gross if prevw is None
               else gross - cost * 0.5 * |w - prevw|.sum())
    prevw = w
```

This is the function the self-test asserts has **bit-identical
top-decile selection** to `heads.hard_top_decile_returns`.

## `_sharpe(r)`

`r.mean() / (r.std(ddof=1) + 1e-12) * sqrt(ANN)`.

## `deflated_sharpe(diff, n_trials=DSR_N)`

Bailey & López de Prado (2014). Inputs: the paired-difference
return series. Returns `(annualised_diff_Sharpe, DSR_prob)` where
`p = 1 - DSR_prob` is the operational p-value.

Internals:

- per-period Sharpe `sr = mean / std`,
- skew `g3` and excess-kurtosis-via-`g4`,
- expected max-Sharpe under N trials via Gumbel approximation:
  `sr0 = sqrt(1/(n-1)) * ((1 - γ) Φ⁻¹(1 - 1/N) + γ Φ⁻¹(1 - 1/(N e)))`,
- adjusted denominator `sqrt(max(1 - g3·sr + (g4 - 1)/4·sr², ε))`,
- `DSR = Φ((sr - sr0) sqrt(n-1) / denom)`.

## `cscv_pbo(perf_matrix, dates)`

PBO via CSCV (López de Prado 2016): S=10 → C(10,5)=252 splits. For
each: pick IS-best by mean train-IS performance; rank it on OOS;
logit-transform the OOS rank; PBO is `mean(logit <= 0)`.

The `perf_matrix` columns are the 3 arms scored (risk / mse /
ridge); rows are per-date performance. The logit-PBO is the
fraction of recombinations where the in-sample-best arm is below
the OOS median — exactly what we want as a primary overfitting
control for the H1 accept.

## `score_h1(risk_scores, mse_scores, ridge_scores, y, did, sym_id)`

THE H1 scorer. All three arms scored on the SAME hard endpoint:

```python
pr = _net_hard_port(risk_scores,  y, did, sym_id)
pm = _net_hard_port(mse_scores,   y, did, sym_id)
pg = _net_hard_port(ridge_scores, y, did, sym_id)
d_vs_mse, d_vs_rdg = pr - pm, pr - pg

for tag, dd in (("vs_mse", d_vs_mse), ("vs_ridge", d_vs_rdg)):
    ds, dp = deflated_sharpe(dd)
    out[f"dSharpe_{tag}"] = ds
    out[f"DSR_{tag}"]      = dp
    out[f"p_{tag}"]        = 1 - dp

out["dRankIC_vs_mse"] = spearmanr(risk_scores, y).statistic
                        - spearmanr(mse_scores, y).statistic
out["PBO"]            = cscv_pbo(per_date_arm_matrix, unique_dates)

out["H1_ACCEPT"] = bool(
    dSharpe_vs_mse   >= 0.20
AND dSharpe_vs_ridge >= 0.20
AND dRankIC_vs_mse   >= 0.01
AND PBO              <= 0.5
AND p_vs_mse         <  0.05
AND p_vs_ridge       <  0.05)
out["_endpoint"] = "hard_top_decile_returns"        # provenance tag
```

## Self-test (closes re-jury-7 FATAL)

`python engine/backtest.py` runs an in-file test that asserts,
**without a trained model**:

- The scorer's call path contains `argsort(-s)` (the hard endpoint)
  and does NOT contain any `soft_top_decile` reference.
- `_net_hard_port` selects the same top-decile rows as
  `heads.hard_top_decile_returns` — parity asserted at
  `max|d| ≤ 3e-18` (machine epsilon).
- Signal arm vs noise arm: SR(risk) > SR(mse) when scores carry
  the signal.
- Identical arms (everything identical) do NOT spuriously accept H1.
- PBO ∈ [0, 1].

Output:
```
[bt] hard-endpoint parity max|d|=3.47e-18
[bt] signal vs noise: SR_risk=… SR_mse=… dSharpe_vs_mse=… PBO=…
     H1_ACCEPT=True
[bt] identical-arms H1_ACCEPT=False (must be False)
BACKTEST-SELFTEST: ALL PASS
```

The self-test runs every time the scorer file is invoked directly,
catching any future regression that would route the scorer through
the soft surrogate.

## The turnover burn-in fix (re-jury-7 MAJOR)

Earlier versions of the backtest charged turnover on the very first
rebalance too. PREREG §4 says the strategy is *deployed once* at the
start, so the initial deployment cost is NOT a recurring cost — and
charging it asymmetrically across the H1 paired difference (where
both arms start at zero weight) introduces a spurious systematic
bias against the arm with higher activity.

Fix: the first `out.append(gross)` is **uncharged** in both
`heads.portfolio_returns` (training) AND `backtest._net_hard_port`
(eval). The burn-in is identical in train and eval so the trained
objective matches the §4-net evaluation rule (k.2).

## Files of record (this stage)

- `engine/backtest.py` (the scorer; self-tested).
- `engine/heads.py` `hard_top_decile_returns` (the canonical
  endpoint).
- PREREG §12 k.2 (surrogate vs endpoint), k.3 (DSR N=1152),
  k.4 (CSCV PBO S=10/252), k.6 (the accept rule).
