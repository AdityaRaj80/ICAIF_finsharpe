# 00 — Project overview

## What this work is

A leakage-controlled, deflation-tested, PBO-tested **methodology
benchmark** on the (contaminated, widely-used) **FNSPID** equity +
news dataset. The single empirical question is whether a
model-agnostic differentiable cross-sectional **portfolio-Sharpe (+
epistemic uncertainty)** training objective changes cross-sectional
predictive skill **relative to** vanilla return-MSE and a Ridge
baseline, across **eight modern architectures**, under a
pre-registered, leakage/deflation/PBO-controlled protocol.

## What this work is NOT

- Not a new fundamental method. The objective itself is the increment;
  the carrying contribution is the **honest finding** (very plausibly:
  deep objectives do not beat Ridge once survivorship/leakage/
  deflation/PBO are respected).
- Not a bias-free economic statement. All absolute Sharpes / dollar
  metrics are appendix-only, labeled "within a disclosed survivor
  universe — levels NOT investable; completeness only."
- Not an external-validity claim across markets. FNSPID-only.

## The single H1

> For a given backbone, the portfolio-Sharpe(+uncertainty) objective
> improves the paired test cross-sectional rank-IC and the Deflated
> Sharpe of the return-difference series vs (a) the same backbone with
> MSE and (b) Ridge on identical features — the single pre-registered
> comparator. ΔDSR ≥ 0.20, Δrank-IC ≥ 0.01, p<0.05 post-PBO, PBO≤0.5,
> at H=5.

A null is the informative answer; k.1 forbids amending the rule to
manufacture a positive.

## Target venue

ICAIF '26 main track (single track, 8-page sigconf, no supplementary,
deadline 2026-08-02). The CFP explicitly invites uncertainty
quantification, model validation, trading / financial forecasting —
this work is in scope.

## The 8 backbones × 2 arms × Ridge

| Backbone | Reference | Role |
|---|---|---|
| iTransformer | Liu et al., ICLR 2024 | variate-token transformer |
| PatchTST | Nie et al., ICLR 2023 | channel-independent patched transformer |
| TFT | Lim et al., 2021 | interpretable attention + VSN |
| GCformer | Wu et al. style | global-decay + local attention |
| DLinear | Zeng et al., AAAI 2023 | canonical strong-simple baseline |
| LSTM | classical | sequential RNN baseline |
| RNN | classical | unit-cell RNN baseline |
| CNN | TCN-style | 10-layer dilated causal conv (RF 2047 ≥ 504) |

Each trained with two heads (arms):
- **mse** — vanilla return-MSE head.
- **risk** — the differentiable soft-top-decile portfolio-Sharpe head,
  composite loss `L = a·(-Sharpe of NET portfolio) + g·MSE + b·NLL`
  with coefficients pinned in PREREG §9.

Plus **Ridge** on identical features (rebalance-row x), closed-form,
the single H1 comparator.

## Current state (as of this writing)

- All preparatory stages (dataset, sentiment, features, engine,
  models, anchors, determinism, backtest scorer, launcher) complete
  and verified.
- PREREG amended through k (final binding amendment) and SHA-stamped.
- PREREG k.1 binding-freeze fired at `2026-05-20T03:22:49Z` on
  `gpunode4.hpc.bits-hyderabad.ac.in`, sha
  `340e68bcc4f4c327bd39d9fb798c6940e9a63e8c8f1bf349bda9ddd140eee08d`.
- H=5 Phase-2 SLURM campaign live: 17 HPO + 1275 evaluation tasks
  + aggregator on BITS HPC, mirrored across H100, A100, V100.
- Documentation in `Reports/` written WHILE the campaign runs.

## Hard constraints maintained throughout

- HPC: never write to `$HOME`; all I/O on
  `/scratch/goyalpoonam/finsharpe/icaif2026/`. Optuna and other
  Python libs installed to `/scratch/.../pylibs` via `pip --target`.
- Shared HPC account `goyalpoonam` — isolate work, do not clobber
  others, throttle SLURM array concurrency to be a good cluster
  citizen.
- Never commit DATA, parquet panels, weights, or PDFs to git
  (gitignored).
- `git push` to `AdityaRaj80/ICAIF_finsharpe` is environment-blocked
  by a data-exfiltration classifier — local commits only; the user
  pushes manually when ready.
- A100 was originally excluded ("A100s are slower than H100/H200,
  use all of them" meant all of H100+H200 — explicitly overridden by
  the user later: "all three GPUs — H100, H200 and A100s, whichever
  is free we will utilize"; and the cluster has no H200 partition).

## Where each topic is documented

See [`README.md`](README.md) for the full file index.
