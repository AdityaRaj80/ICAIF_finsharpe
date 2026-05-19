# Reference — Reusable Formulas, Expressions & Specs

Technical content extracted for reuse: losses, prediction heads, feature set,
schedules, and training hyperparameters. Math/specs only.

---

## 1. Prediction heads / dual-arm design

Per `(model, horizon, fold)` cell, two arms are trained:

**MSE arm (baseline)** — `MSEReturnHead`:
```
y_close_hat = backbone(x)                      # [B, pred_len], z-scored close
y_ret_hat   = Linear(pred_len -> 1)(y_close_hat)   # [B] scalar log-return
L_mse       = mean( (y_ret_hat - y_logret)^2 )
```

**Risk-aware arm** — `RiskAwareHead`: same `return_head`, plus two auxiliary
MLPs over the last 20 input rows:
- `sigma_head -> log sigma^2_H`, bias-initialised to `log(sigma_typ^2)` with
  `sigma_typ = 0.01 * sqrt(H)` (so the tanh argument is non-trivial at init)
- `vol_head  -> predicted forward log realized vol`
- EMA-tracked gate thresholds `tau_sigma, tau_v` stored in the head (so they
  freeze under `eval()` and persist in the checkpoint)

---

## 2. Composite risk-aware loss (operates entirely in return space)

5-term composite:
```
L = alpha * L_SR_gated     # gated differentiable Sharpe (primary objective)
  + beta  * L_NLL          # heteroscedastic Gaussian NLL on the H-step return
  + gamma * L_MSE_R        # return-MSE anchor (prevents degenerate flat collapse)
  + delta * L_VOL          # MSE on log realized-vol target (calibrates vol head)
  + eta   * L_GATE_BCE     # BCE: confidence gate vs. realized profitability
```
Fixed coefficients: `beta = 0.5`, `delta = 0.3`, `eta = 0.1`.
Scheduled coefficients: `alpha`, `gamma` (see §4).

### 2.1 Gated differentiable Sharpe — `L_SR_gated`
```
position   = tanh( alpha_pos * mu_ret / (sigma + eps) )          # alpha_pos = 10
gate       = sigmoid((tau_sigma - sigma)/(s_sigma * T))
             * sigmoid((tau_v - log_vol)/(s_v * T))
strat_r    = gate * position * true_return
L_SR_gated = - mean(strat_r) / ( std(strat_r) + eps )            # negative Sharpe
```
- Argument is `mu/sigma` (**Sharpe units**), deliberately NOT Kelly `mu/sigma^2`
  — more robust under a *learned*, noisy sigma estimator.
- Optional cross-sectional path (B1): build K synthetic cross-sections per
  batch, compute Sharpe of long-short leg-normalised portfolio returns
  (long-short generalisation of Zhang–Zohren–Roberts 2020).

### 2.2 Heteroscedastic Gaussian NLL — `L_NLL`
Network predicts both `mu_return_H` and `log sigma^2_H`:
```
L_NLL = 0.5 * ( log sigma^2 + (y - mu)^2 / sigma^2 )
```

### 2.3 Return-MSE anchor — `L_MSE_R`
Plain MSE on predicted vs realized H-step return. Without it the Sharpe term
can collapse to a degenerate flat predictor.

### 2.4 Volatility calibration — `L_VOL`
MSE of predicted forward log realized vol vs a realized-vol target.

### 2.5 Confidence-gate BCE — `L_GATE_BCE`
Binary target = realized profitability `1{ position * return > 0 }`.
- End-to-end, network-internal variant of meta-labeling (López de Prado,
  *Advances in Financial ML* 2018, Ch.3 §3.6): same "was the trade profitable"
  label, but trained jointly with the mu/sigma/vol heads in one backward pass;
  the gate is structurally a product of sigmoids over the network's own
  uncertainty estimates.
- With `bce_use_margin = True` (default): only supervise "confident" samples
  whose `|P&L|` is above the within-batch median.

---

## 3. Gate temperature schedule
```
T = max( 0.13, 1.0 * 0.92^epoch )
```
Anneals from soft-attention (early, T=1.0) to near-binary kill-switch
(late, T=0.13).

---

## 4. (alpha, gamma) phase schedule (`step_epoch`)

| Phase | Epochs | alpha (Sharpe) | gamma (MSE anchor) | Intent |
|---|---|---|---|---|
| P1 warm-up | 0 – 7  | 0.0 | 1.0 | pure return-MSE, stabilise mu |
| P2 ramp    | 8 – 24 | 0.3 | 0.7 | introduce Sharpe gradient |
| P3 Sharpe  | >= 25  | 0.7 | 0.5 | Sharpe-dominant; anchor never < (1/2)*alpha |

---

## 5. Alpha158-lite feature set (`enc_in = 69`)

6 raw + ~63 Qlib Alpha158-lite factors. All factors are **scale-invariant**
(ratios, log-returns, normalised ranks) so they are comparable across stocks
before per-stock z-scoring. `CLOSE_IDX = 3`.

- **6 raw:** `Open, High, Low, Close, Volume, scaled_sentiment`
  (OHLC are Adj-Close-adjusted; Volume is `log1p`'d after alpha computation)
- **K-line shape (7):** `KMID, KLEN, KMID2, KUP, KLOW, KSFT, KSFT2`
- **Multi-lag log returns:** `RET_{5,10,20,30,60}`
- **Rolling stats over {5,10,20,30,60}:** `MA, STD, MAX, MIN, RANK, HL` ratios
- **Volume rolling over {5,10,20,30,60}:** `VMA, VSTD`
- **Volume–price correlation:** `CORR_{10,20,60}`
- **Momentum:** `MOM_{5,10,20}`
- **Sentiment:** `SENT_DELTA, SENT_MA_{5,20}, SENT_STD_{5,20}`

Reference: Qlib Alpha158 (Yang et al. 2020, arXiv:2009.11189).

---

## 6. Sentiment aggregation
Daily sentiment uses **exponential-decay** aggregation: on days with no news,
carry the last score forward with exponential decay, so the score stays
continuous across the (sparse) news calendar.

---

## 7. Training recipe / hyperparameters

```
method        = global         # one model over all stocks, calendar split
epochs        = 80
lr            = 3e-4
lr_min        = 3e-6
lradj         = cosine
warmup_epochs = 5
patience      = 20             # early-stop on validation rank-IC (NOT val MSE)
amp           = on             # bf16/fp16 autocast
batch_size    = 256            # 512 OOMs GCFormer (69 ch); 128 too slow
SEQ_LEN       = 504            # 2 trading-year look-back
```
- Model-selection / early-stop metric = **validation rank-IC** (aligned with
  the cross-sectional trading objective), not val MSE.
- Horizons used: `H in {5, 20, 60}`. `H in {120, 240}` were dropped previously
  because the validation set was empty after embargo in 1-year folds — needs a
  CV scheme that supports long horizons.
- Prior walk-forward folds: F1 test'20, F2 test'21, F3 test'22, F4 test'23.

---

## 8. Literature anchors (tied to the above)
- **Moody & Saffell** — differentiable Sharpe / direct reinforcement;
  foundational for `L_SR`.
- **Zhang, Zohren & Roberts 2020**, "Deep Learning for Portfolio
  Optimization," arXiv:2005.13665 — Sharpe-loss portfolio; the B1
  cross-sectional path generalises it to long-short.
- **López de Prado, *Advances in Financial Machine Learning* (2018)** —
  meta-labeling (Ch.3 §3.6), basis for `L_GATE_BCE`.
- **Yang et al. 2020, Qlib**, arXiv:2009.11189 — Alpha158 feature set.
