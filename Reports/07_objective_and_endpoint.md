# 07 — Training objective and the §4 evaluation endpoint

This is the most consequential piece of the work — both as a
contribution and as the source of the original FATAL-1.

## The §4 evaluation endpoint (target, hard, non-differentiable)

The PREREG §4 endpoint is the **long-only top-decile equal-weight**
portfolio rebalanced on every H=5 trading days, evaluated on the
non-overlapping H-spaced rebalance dates of the test split:

- At each rebalance date d, take the model's scores across the
  cross-section.
- Select the top 10% by score (`FRAC=0.10`).
- Equal-weight them.
- Hold for H=5 trading days; the realized return is the period
  return.
- Subtract one-way turnover cost `COST=0.001` (10 bps) on the L1
  weight change vs the previous rebalance; the FIRST rebalance
  is UNCHARGED (burn-in / deploy-once).

This is the rule that scoring uses, in `engine/heads.py`
`hard_top_decile_returns` and `engine/backtest.py` `_net_hard_port`.
It is non-differentiable (topk is non-differentiable) and so cannot
be optimised directly by gradient descent.

## FATAL-1 — the original objective was NOT the endpoint

The first implementation of the "Sharpe-aware" head used
`softmax(mu / τ)` as the cross-sectional weight rule. Re-jury-5
caught this as mathematically false:

- `softmax` has **full support** over all stocks; weights never go
  to zero.
- At realistic `mu` scale (returns ~ a few percent, τ at any
  reasonable value), the softmax is near-uniform across the
  cross-section.
- The claim that this was "the image of §4" was therefore false.
  The training objective and the evaluation objective were
  optimising different things.

This was rated **FATAL** because the paper's H1 is precisely about
whether optimising the cross-sectional Sharpe objective changes
predictive skill — if the objective was not the cross-sectional
Sharpe objective at all, every H1 claim collapses.

## The differentiable soft-top-decile (FATAL-1 fix)

`engine/heads.py` `soft_top_decile(score, did, tau, frac=FRAC)`:

```
per rebalance date d, over the cross-section of scores s:
  q   = quantile(s, 1 - frac)          # the decile cutoff (detached)
  sd  = std(s)                         # scale (detached) -> scale-aware
  w_i = sigmoid((s_i - q) / (tau * sd))   # long-only, w >= 0
  w   = w / sum(w)                     # budget-normalized (sum = 1)
tau -> 0   ==>   w -> uniform over the top `frac` names == §4 endpoint
```

Properties (and why this is the right surrogate):

- **Long-only** by construction (sigmoid > 0).
- **Budget-normalized** by construction (sum to 1).
- **Differentiable** for τ > 0.
- **Scale-aware**: dividing by `sd` removes sensitivity to the
  arbitrary scale of `mu`.
- **τ→0 limit equals the §4 hard endpoint**. This is asserted in
  `test_pipeline.py` T4: as τ shrinks, the soft weights converge to
  uniform-over-top-decile and the soft-endpoint Sharpe converges to
  the hard-endpoint Sharpe.

τ has **no default** in the function signature — must be passed
explicitly. It is **pinned** in PREREG §9 at `0.05` for the campaign.

## AMP-quantile fp16 fix

When the real-path determinism test ran for the first time it
crashed inside `soft_top_decile`: `torch.quantile` rejects fp16
inputs, and weight normalisation is numerically unstable in half
precision under autocast. Fix: cast `score = score.float()` at the
top of `soft_top_decile` (an upcast inside autocast is safe; gradients
to the encoder remain fp32). Re-jury-6 MAJOR closed.

## `composite_risk_loss`

`engine/heads.py` `composite_risk_loss(z, head, batch, tau,
cost=COST, a=0.7, g=0.5, b=0.5)`:

```
L = a * ( - Sharpe of NET soft-top-decile portfolio over the batch's
                   H-spaced dates )
  + g * MSE(mu, y)                              [anchor]
  + b * NLL(mu, sigma2, y)                      [sigma calibration]
```

Coefficients, τ, cost are **PINNED** in PREREG §9 (CI-asserted by
`check_prereg_constants.py`). Sharpe std is **unbiased**.

### Why NET of L1 turnover (Task 50)

`portfolio_returns(...)` builds the per-date NET return as
`gross - cost * 0.5 * |w - w_prev|.sum()`, with the FIRST rebalance
uncharged (burn-in). The previous weight is **detached** so the
model is incentivised to *reduce* turnover, not to game the gradient
through the previous step. This makes the trained objective match
PREREG §4-net (re-jury-5 MAJOR). `sym_id` is plumbed through the
loader so weights can be aligned by symbol across consecutive
rebalances.

### Why NLL is ABLATION-ONLY, not used in allocation

The original "Risk-Aware" head also predicted a `sigma` and the
allocation was supposed to be uncertainty-aware. Re-jury caught
that the gate (uncertainty → weight) was unused; training a head
that did nothing for H1 was an over-claim. The slim version:

- `mu` is the score that is *ranked* into the soft-top-decile.
- `sigma` (from `logvar`) is only an **optional NLL calibration**
  ablation — it does NOT enter the allocation. The paper is honest
  about this: the head learns a calibrated uncertainty but does not
  use it in the portfolio; whether using it would help is a
  documented future direction.

## k.2 — surrogate ≠ endpoint (dissolves re-jury-6 FATAL)

A late re-jury asked: "you train one estimand (the soft-top-decile
surrogate's Sharpe) and test another (the hard endpoint). Isn't
that a model misspecification problem?" Amendment **k.2** answers:

- They are *intentionally* different (a differentiable surrogate
  vs the hard non-differentiable evaluation rule).
- The paper makes **no claim** that the surrogate equals the
  endpoint; whether optimising the surrogate moves the hard
  endpoint is precisely the empirical question H1 asks.
- Only the hard endpoint is reported. Scoring is exclusively via
  `engine/backtest.py` `score_h1 → hard_top_decile_returns`. The
  self-test asserts no `soft_top_decile` reference appears anywhere
  in the scorer's call path.

τ MAY be annealed via `heads.tau_schedule` (`0.10 → 1e-3` cosine)
to push the trained surrogate toward the hard endpoint as training
progresses. Validity does NOT depend on annealing — static τ=0.05
is fully compliant.

## What this stage gave up vs the original sketch

- The "uncertainty-aware allocation" overclaim. Replaced with an
  honest "NLL calibration ablation, not used in allocation."
- Any pretence that the surrogate is the endpoint. They are
  separate objects with separate roles; H1 is precisely the
  question of whether changes in one move the other.
- A naïve "phase schedule / α_pos 10 / β δ η" mechanism that
  appeared in early PREREG §9 but never existed in code; deleted in
  k (re-jury-5).

## Files of record (this stage)

- `engine/heads.py` (the head and the loss).
- `engine/backtest.py` (the scorer).
- `engine/test_pipeline.py` (T4 = soft→hard limit; ALL PASS).
- PREREG §9 + k.2 (the binding spec).
