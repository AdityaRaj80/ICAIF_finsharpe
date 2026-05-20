# 10 — Determinism

The PREREG (§7/§8) requires each of the 5 seeds {0..4} to be
**bit-reproducible**. This is a non-trivial property under AMP,
GradScaler, cuDNN, dropout, and the date-grouped loader; achieving
it required a hardening pass + a real-path proof.

## The seed harness — `engine/determinism.py`

`seed_all(seed)`:

```python
os.environ["PYTHONHASHSEED"] = str(seed)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)        # warn_only on cpu-only build
g = torch.Generator(); g.manual_seed(seed)
return g                                        # for DataLoader(generator=g)
```

Critical detail: `CUBLAS_WORKSPACE_CONFIG=:4096:8` MUST be set
**before CUDA is initialised**. The harness sets it via
`os.environ.setdefault` at import time, so any subsequent
`torch.cuda.*` call respects it.

The `engine/determinism.py` toy 2-run check uses iTransformer
forward + composite loss + backward; it asserts:

- run(seed=0) == run(seed=0) (bit-identical),
- run(seed=0) != run(seed=1) (the seed matters).

PASS.

## The real-path proof — `engine/determinism_real.py` (re-jury-6 MAJOR)

Re-jury-6 ruled the toy test insufficient: the Phase-2 training path
uses
**DateGroupedLoader + AMP autocast + GradScaler + Adam +
`composite_risk_loss` over multiple epochs**, all of which can
break bit-identical reproducibility for subtle reasons. The
real-path test exercises that exact stack:

- Symbols: first 30 from `universe/tier1.txt` (kept small for
  walltime; the determinism property does not depend on universe
  size).
- Model: iTransformer (transformer family) and CNN (conv family)
  — two architectures with different cuDNN paths.
- Two epochs of real loader batches.
- AMP autocast + GradScaler enabled (`enabled=(DEV=="cuda")`).
- `composite_risk_loss` with the pinned PREREG-§9 constants.

After two runs at seed 0, asserts:
```
max  |state_dict_a[k] - state_dict_b[k]|  ==  0.0  (bit-identical)
```
on every floating-point parameter.

Report: `bench/determinism_real_report.md` — PASS for both
backbones.

## The AMP-quantile fp16 bug found by this proof

The real-path test crashed on its first run inside
`engine/heads.py` `soft_top_decile`. Root cause:

- AMP autocast was downcasting the cross-section scores to fp16.
- `torch.quantile` rejects fp16.
- Even if it had accepted, the weight-normalisation step (`wi.sum()`)
  is numerically unstable in fp16 — different summation orders
  produce different ULPs, breaking bit-identical.

Fix (in `engine/heads.py`):
```python
score = score.float()    # AMP-safe upcast inside autocast region
```

This is safe: the upcast is local; encoder gradients remain fp32;
the surrogate weight computation is deterministic. After the fix,
both 2-run checks bit-identical at 0.0.

Without this fix the entire 1275-task Phase-2 campaign would have
non-reproducible weights — making the 5-seed variance estimate
meaningless. Catching it at the determinism gate, before launch,
was exactly the point of having that gate.

## Why bit-identical matters here

The PREREG accepts "5-seed variance reduction" only if each seed is
its own reproducible experiment. If seeds drift run-to-run, the
across-seed dispersion conflates non-determinism noise with
genuine variance and the across-seed median is no longer the
fixed-config estimator k.3 says it is.

## Limits of this guarantee

- Bit-identical is per-(seed, hardware). Different GPU SKUs can
  produce different floating-point outputs even with all the above
  flags. The campaign mirrors across H100/A100/V100, so the
  per-cell output parquet is hardware-tagged in scratch logs. For
  the H1 verdict this is fine because aggregation is over (date,
  sym) scores; per-cell hardware can differ.
- The race condition where two SLURM array mirrors try the same
  cell simultaneously: the idempotent skip in `phase2.run_cell`
  prevents most of this, and the rare window is bounded — both
  would produce identical scores if seeded identically on
  identical hardware. With different hardware mirrors this is the
  only place where the campaign tolerates non-determinism across
  arrays; for k.3 purposes the OOF score is whichever finished
  first. This is disclosed.

## Files of record (this stage)

- `engine/determinism.py`, `engine/determinism_real.py`.
- `bench/determinism_real_report.md`,
  `bench/_rejury6_run.log`.
- PREREG §7 / §8 (the determinism requirement).
