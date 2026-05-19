# ARCHITECTURE_REVIEW — backbones vs canonical papers & pipeline fit

Cross-verification of `engine/models.py` against each canonical source AND
against OUR task. **Our task is NOT standard long-horizon forecasting.** It
is **cross-sectional scalar prediction**: per (stock, day) a lookback
`x∈ℝ^{504×65}` → one scalar = forward **H-day log-return**; per day, rank
stocks cross-sectionally → portfolio; H ∈ {5,20,63,126,252} (H=5 the only
inferential horizon, PREREG §3). Two arms share the encoder: MSE head vs
risk head (μ, logσ², vol, gate) with the 5-term composite (Reference §2).

## Per-model verification

| Model | Canonical (verified) | Core we implement | Deviation | Pipeline-fit |
|---|---|---|---|---|
| **iTransformer** | Liu+ ICLR'24, arXiv:2310.06625, thuml/iTransformer | variate-as-token: embed each feature's full length-L series, attention across the 65 variates, FFN/var | none in core; no temporal attn (as in paper) | OK — encoder→scalar head valid; 504 cheap (no temporal quadratic) |
| **PatchTST** | Nie+ ICLR'23, arXiv:2211.14730, yuqinie98 | channel-independent patching + shared Transformer | **+late cross-channel mean-fuse** (canonical never mixes channels) — needed for a single scalar | ⚠ **M2**: documented deviation; ablate fuse=mean vs linear; report it |
| **TFT** | Lim+ 2021 IJF | per-feature GRN + variable-selection softmax + LSTM + interpretable MHA | **observed-only**: no static / known-future streams | ⚠ **M4**: FNSPID has none; reduced-TFT is defensible but state it; lose TFT's headline interpretability claim |
| **GCformer** | Zhao+ **CIKM'23**, arXiv:2306.08325 | global depthwise long-kernel conv branch + local Transformer branch → fuse | **compact approx** of the *structured* global kernel (3-param parameterization) | ⚠ **M5**: validate vs official before frozen run; approximation may change long-range capacity |
| **DLinear** | Zeng+ AAAI'23 Oral, cure-lab/LTSF-Linear | moving-avg decomposition + per-feature linear over lookback | maps to representation + mean-pool features (not L→pred_len) | OK — intended strong linear reference |
| **LSTM** | std baseline | 2-layer LSTM, last hidden | none | ⚠ **M7**: 504 steps → vanishing gradient; kept as honest weak baseline, NOT crippled — disclose so a reviewer doesn't read underperformance as a bug |
| **RNN** | std baseline | 2-layer Elman RNN | none | ⚠ **M7** (worse than LSTM at 504) — same disclosure |
| **CNN/TCN** | Bai+ 2018 | dilated causal conv, dilations 1..256, k=3 | none | OK — receptive field 1+2·Σdil = **1023 ≥ 504** (verified covers lookback) |

Smoke test (`engine/registry.py`): all 8 × {mse,risk} construct, produce
`[N,128]`, finite loss, `backward()` OK (0.06–0.81M params @ d=128 smoke;
true capacity set at train per Reference §7 / PREREG).

## Global pipeline-fit findings (the load-bearing ones)

- **P1 — encoder-pooling is a deliberate adaptation.** Every backbone is
  used as an *encoder → pooled rep → scalar head*. None of these papers
  was designed for cross-sectional single-scalar equity return prediction;
  this is a legitimate, standard adaptation for a benchmark but MUST be
  stated as the harness choice (uniform head, identical across models — the
  point is the controlled contrast, not SOTA per model).
- **P2 — what each model predicts (all horizons).** Exactly one scalar per
  (stock,day): forward H-day log-return. A SEPARATE model is trained per H
  ∈ {5,20,63,126,252}. Only H=5 carries H1 inference (n_eff≈150); H≥20 are
  descriptive (PREREG §3). The vol/uncertainty head predicts forward
  H-realized-vol + a gate; epistemic uncertainty = predictive σ (MC-dropout
  spread optional at eval).
- **P3 — CROSS-SECTIONAL SHARPE REQUIRES DATE-GROUPED BATCHES (critical).**
  `composite_risk_loss` groups by `batch['date_id']` and computes Sharpe
  over per-date portfolio returns. The Phase-2 dataloader MUST emit
  date-grouped batches (all stocks for a set of rebalance dates), NOT
  random (stock,window) pairs, or the portfolio-Sharpe term is undefined.
  This constrains batching, embargo (purge by date across the group), and
  the H-spaced non-overlapping schedule (PREREG §3/§4). **Single biggest
  correctness dependency for the risk arm.**
- **P4 — HPO/identical-harness parity.** Same d_model grid, optimizer,
  early-stop (nested inner-CPCV rank-IC), N_HPO=32, seeds {0..4} for ALL 8
  + Ridge (PREREG §6/§7). Encoder pooling identical across models.
- **P5 — leakage interface.** Features already causal & train-only-z
  (features_leakage_qc 8/8). Models add no leakage IF the date-grouped
  loader respects purge+embargo=H around fold/test boundaries (P3).
- **P6 — paired endpoint.** H1 = paired Δ(risk-arm − mse-arm) and vs Ridge;
  residual survivorship-interaction bias disclosed (PREREG hdr/§1), not
  cancelled.

## Validation TODO before the frozen Phase-2 run
1. Validate iTransformer / PatchTST / TFT / GCformer compact cores against
   their official repos on a public benchmark (ETTh1) — match trend/ballpark,
   else swap in official modules. (M2/M4/M5.)
2. Implement & unit-test the **date-grouped, purged, H-spaced** dataloader
   (P3) — this is a prerequisite, not optional.
3. Confirm TCN causal trim is exactly causal (no right-context leak).
4. Decide & freeze: d_model grid, PatchTST fuse, TFT observed-only — log
   in PREREG §9 before training.
