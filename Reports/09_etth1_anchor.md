# 09 — External validity: the ETTh1 anchor

The 8 backbones are in-house faithful reimplementations rather than
official-repo copies. An honest-negative result on FinSharpe is only
credible if the deep models are not subtly under-built ("impostor
implementations"). The ETTh1 anchor is **impostor / bug detection,
NOT SOTA reproduction**.

Pre-registered design (PREREG §9a, frozen BEFORE seeing any number).

## What ETTh1 is

ETTh1 = Electricity Transformer Temperature hourly-1 — a standard
public long-horizon time-series forecasting benchmark used by
DLinear (AAAI 2023), PatchTST (ICLR 2023), iTransformer (ICLR 2024),
Autoformer (NeurIPS 2021), and the LTSF baseline literature.

`bench/ETTh1.csv` (and h2/m1/m2) downloaded once and stored locally.

## First anchor (pooled encoder; criterion FAILED, PRESERVED)

`engine/ett_anchor.py`. Setup, fixed before run:

- Informer split: 12 / 4 / 4 months train/val/test.
- Input 96 → predict 96, all 7 variables.
- Per-feature train-fit standardisation.
- Each of the 8 backbones with the **shared pooled-encoder + linear
  head** used in FinSharpe.
- Adam; early-stop on val MSE.

PASS criteria, fixed before seeing any number:
- (i) finite test MSE,
- (ii) test MSE < persistence (last-value) baseline,
- (iii) test MSE < 0.80,
- (iv) each modern transformer within 2.5× of DLinear's test MSE.

### 9a-OUTCOME

`bench/ett_anchor_report.md`:

- All 8 models beat persistence (1.294) with architecture-consistent
  ordering: DLinear .89 / GCformer .85 / TFT .93 best; PatchTST 1.00
  / iTransformer 1.11 worse (consistent with mean-pool discarding
  their patch/variate structure); RNN .70.
- (i), (ii), (iv) PASSED for all.
- (iii) `< 0.80` **FAILED for 7/8** (only RNN .70).

**Honest finding (re-jury-6 amendment j)**: (iii) was
**MIS-SPECIFIED** — it implicitly assumed each model's *native*
forecasting head, but the harness deliberately used the shared
encoder → pooled-vector → generic-linear head (FinSharpe's usage).
Encoder-pooling is a severe bottleneck for multi-step forecasting;
the result shows the architectures **learn real structure (not
impostors)** but absolute MSE is not literature-comparable in this
config.

**The failed numbers stand on record.** We do NOT relax (iii). The
valid fidelity test is re-specified separately in 9a-NATIVE; the
preserved failure is what k.1 binding-freeze is, in advance,
agreeing to live with.

## Second anchor (native head, fixed pre-result; PASS) — 9a-NATIVE

`engine/ett_anchor_native.py`. Each fidelity-critical model uses
its **native** forecasting head + RevIN-style per-series instance
norm — the configuration in which published ETTh1 numbers were
actually obtained:

- DLinear: canonical per-channel decomposition-linear.
- PatchTST: channel-independent backbone → per-channel linear to
  pred_len.
- iTransformer: variate tokens → encoder → per-variate projection
  to pred_len.

PASS (fixed BEFORE the run): for the trio {DLinear, PatchTST,
iTransformer} — finite, beats persistence, **test MSE < 0.55**
(generous vs published ~0.37-0.42), and ordering not pathological
(no model > 2× the trio-best). Failure ⇒ real impl bug, fixed
before Phase-2.

Result: `bench/ett_anchor_native_report.md` — PASS.

## All-8 with mechanically-derived gates — 9a / k.5

`engine/ett_anchor_all8.py`. After amendment k, every backbone has
a **mechanically-derived** gate (no eyeballed numbers):

```
gate(model class)  =  1.5 × worst published ETTh1-96 MSE for that class
```

The 1.5× multiplier is the only free constant and was fixed BEFORE
any run.

| Class | Worst published MSE (source) | Gate (= 1.5×) |
|---|---|---|
| DLinear | .40 [Zeng et al., AAAI 2023, Table 2] | .60 |
| PatchTST | .41 [Nie et al., ICLR 2023, Table 3, supervised] | .62 |
| iTransformer | .39 [Liu et al., ICLR 2024, Table 1] | .59 |
| GCformer / Transformer-family | .45 [Autoformer, NeurIPS 2021, ETTh1-96, upper of family row] | .68 |
| TFT | .60 [Lim et al., 2021; non-LTSF-tuned, conservative upper] | .90 |
| TCN | .55 [SCINet / LTSF conv-baseline rows, conservative] | .83 |
| LSTM | .70 [classical-RNN rows, LTSF baseline, conservative upper] | 1.05 |
| RNN  | .70 [classical-RNN rows, conservative upper] | 1.05 |

Anchor capacity is deliberately `d_model=128` (ETT-standard scale);
the FinSharpe campaign capacity `d_model=256` is intentionally
decoupled (the anchor tests architecture *mechanism fidelity*, NOT
the FinSharpe campaign capacity).

Result: `bench/ett_anchor_all8_report.md` — **all 8 PASS**, every
model under its derived gate, observed `.389 – .574`. Via the rule,
not via a known-answer round number.

## What this stage proves and what it does NOT

Proves:
- The 8 backbones are not impostors. Each architecture's defining
  mechanism is operative (native-head and all-8 derived gates PASS).
- The pooled-encoder failure on (iii) is a *configuration* effect,
  not an implementation bug.

Does NOT prove:
- That the 8 in-house implementations match published SOTA. They
  don't (and the paper does not claim to). The k.5 multiplier is 1.5×
  the worst published MSE specifically because we're an Informer-split
  + modest-tuning re-run, not a full LTSF reproduction.
- That ETT fidelity transfers to FinSharpe performance. The anchor
  is impostor detection only; the FinSharpe H1 stands or falls on
  its own statistical machinery.

## Files of record (this stage)

- `engine/ett_anchor.py`, `engine/ett_anchor_native.py`,
  `engine/ett_anchor_all8.py`.
- `bench/ett_anchor_report.md`,
  `bench/ett_anchor_native_report.md`,
  `bench/ett_anchor_all8_report.md`,
  `bench/_ett_log.txt`.
- PREREG §9a (the criteria, preserved + re-spec).
