# 06 — Engine and the 8 backbones

## Engine layout

```
engine/
  dataset.py                  # DateGroupedLoader, cpcv_folds, nested_inner, cscv_folds
  heads.py                    # MSEReturnHead, RiskAwareHead, soft_top_decile,
                              # hard_top_decile_returns, portfolio_returns, composite_risk_loss
  models.py                   # the 8 backbones + REGISTRY
  registry.py                 # Model = encoder + head wrapper
  verify_models.py            # defining-property tests per architecture
  test_pipeline.py            # T1-T6 integration tests
  test_cpcv.py                # CPCV fold tests
  check_prereg_constants.py   # CI guard: imports code, fails if PREREG-pinned values diverge
  determinism.py              # seed_all + toy 2-run bit-identical
  determinism_real.py         # real-path 2-run bit-identical (re-jury-6 MAJOR)
  ett_anchor.py               # pooled-encoder ETTh1 anchor (criterion-iii FAILED, preserved)
  ett_anchor_native.py        # native-head re-spec (PASS)
  ett_anchor_all8.py          # all-8 mechanically-derived gates (k.5; PASS)
  backtest.py                 # the executable §4 H1 scorer (hard endpoint, score_h1)
  freeze.py                   # PREREG k.1 binding-freeze
  phase2.py                   # the Phase-2 driver
  phase2_aggregate.py         # H1 verdict aggregator
```

## The 8 backbones

All 8 backbones are **faithful in-house reimplementations**, not
copies of any single official repo (the user explicitly authorized
this — "in-house faithful reimplementations OK"). They share the
contract `forward(x: [N, L=504, F=65]) -> z: [N, d_model=256]` so
the same head and the same training loop can be applied to all of
them (the controlled-contrast principle).

### iTransformer

- Reference: Liu et al., *iTransformer: Inverted Transformers Are
  Effective for Time Series Forecasting*, ICLR 2024.
- Defining property: variates are tokens, not time steps; attention
  is across the F=65 variates, not across L=504 time steps.
- This-repo capacity: `heads=8, layers=4`, d_model=256.
- Verify gate (defining-property): destroying inter-variate
  attention degrades the loss; destroying inter-time attention does
  not (because there isn't any).

### PatchTST

- Reference: Nie et al., *A Time Series is Worth 64 Words*, ICLR 2023.
- Defining property: **channel-independent** backbone; the time
  series is **patched** (length-16 patches, stride 8); a
  permutation-invariant channel-mean head pools the per-channel
  outputs to the cross-section embedding.
- This-repo capacity: `patch=16, stride=8, heads=8, layers=4`,
  d_model=256.

### TFT

- Reference: Lim et al., *Temporal Fusion Transformer*, 2021.
- Defining property: per-variable embeddings + Variable Selection
  Networks (VSN) + Gated Residual Networks (GRN) + LSTM + gated
  multi-head attention (observed-only).
- This-repo capacity: `heads=4`; strength via width 256.

### GCformer

- Defining property: long-memory **global** structured kernels +
  **local** Transformer attention. K=24 decay bases with poles
  `logit(linspace(0.90, 0.9995, K))`; local window 128; local
  Transformer depth 3.
- Long-memory init was caught and fixed during verification (the
  initial implementation's "global" kernel was effectively local).

### DLinear

- Reference: Zeng et al., *Are Transformers Effective for Time
  Series Forecasting?*, AAAI 2023.
- Defining property: canonical seasonal+trend decomposition (moving
  average kernel=25) followed by per-channel **linear** projection.
- Deliberately **kept canonical**: its strength is being the correct
  strong-SIMPLE baseline.

### LSTM

- Defining property: stacked LSTM (3 layers) over the time axis;
  last-step hidden → linear → d_model=256.

### RNN

- Defining property: unit-cell vanilla RNN (3 layers); same shape
  contract as LSTM.

### CNN (TCN-style)

- Defining property: 10 dilated causal Conv1d layers with dilations
  2^0..2^9 and kernel=3 → receptive field 2047 ≥ 504 (full
  sequence). The "TCN" naming matches Bai/Kolter/Koltun usage.

## REGISTRY and `Model` wrapper

`engine/models.py`:
```python
REGISTRY = {"itransformer": iTransformer, "patchtst": PatchTST,
            "tft": TFT, "gcformer": GCformer, "dlinear": DLinear,
            "lstm": LSTM, "rnn": RNN, "cnn": CNN}
```

`engine/registry.py` `Model(name, arm, n_feat, seq_len, d_model)`
glues an encoder to a head:
- `arm == "mse"` → `MSEReturnHead` (single linear → 1).
- `arm == "risk"` → `RiskAwareHead` (`mu` and `logvar` heads;
  logvar clamped to [-10, 4]; dropout 0.1).

`Model.loss(batch)`:
- MSE arm → `F.mse_loss(self.head(z), batch["y_ret"])`.
- Risk arm → `composite_risk_loss(z, head, batch, tau=TAU=0.05)`.

## Why d_model=256

The user instruction: "don't make the models lightweight, make them
strong." Re-jury also flagged: an honest-negative is only credible if
the deep models are well-built; under-tuning would manufacture the
negative. Decision (PREREG §9): uniform training capacity
**d_model=256** for the Phase-2 campaign. The CI guard
`check_prereg_constants.py` asserts this number is what the code
imports — no silent drift.

## Verification — `verify_models.py` and the FATAL-4 fix

The first version of `verify_models.py` had gameable tests (e.g.,
"the module name contains 'patch'"). Jury Round 5 flagged this as
**FATAL-4** ("an impostor implementation could pass"). The rewrite
uses **defining-property checks**:

- iTransformer: ablate the per-variate attention and assert loss
  degrades materially; ablate per-step attention and assert no
  change (because there should be none).
- PatchTST: assert channel-independence (permuting input channels
  permutes outputs identically) and patch-stride compliance.
- TFT: assert VSN gating is differentiable and active.
- GCformer: assert the structured global kernel has long-memory
  poles in the expected range.
- DLinear: assert the decomposition kernel matches kernel=25.
- LSTM/RNN/CNN: receptive-field and parameter-count gates.

Result: `verify_models.py` ALL PASS on the corrected engine.

## Strengthening pass (Task 51)

After FATAL-4 was fixed, the 8 backbones were strengthened to the
research-grade capacity above. PREREG §9 was re-synced to the new
constants; CI guard re-asserted. Verify ALL PASS.

## Test stack

- `test_pipeline.py` T1-T6: loader integration, batch shapes,
  per-symbol non-overlap, soft→hard limit equality, NLL behaviour,
  end-to-end loss differentiability. ALL PASS.
- `test_cpcv.py`: CPCV fold count, purge correctness, train/test
  partition disjointness. ALL PASS.
- `check_prereg_constants.py`: 35 constants asserted from code; PASS.
- `verify_models.py`: defining-property gates; ALL PASS.
- `backtest.py` self-test: hard-endpoint parity 3e-18; signal arm
  beats noise arm; identical arms do NOT accept H1; PBO in [0,1].
  ALL PASS.
- `determinism_real.py`: real-path 2-run bit-identical on
  iTransformer + cnn. PASS.

## Files of record (this stage)

- `engine/*.py` (above).
- `bench/ett_anchor*report.md`,
  `bench/determinism_real_report.md`.
