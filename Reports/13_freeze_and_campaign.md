# 13 — The freeze and the live H=5 campaign

## The moment of the binding-freeze

PREREG k.1 fired on the BITS HPC at
**`2026-05-20T03:22:49Z`**, on node
**`gpunode4.hpc.bits-hyderabad.ac.in`** (A100 80GB), during the
first Phase-2 task. The stamp:

```
340e68bcc4f4c327bd39d9fb798c6940e9a63e8c8f1bf349bda9ddd140eee08d  \
  frozen_utc=2026-05-20T03:22:49.872098Z  \
  host=gpunode4.hpc.bits-hyderabad.ac.in  \
  job=phase2-hpo
```

This file (`FREEZE_STAMP`) is committed to git locally (commit
`9aaf044`) as immutable provenance. The recorded SHA in
`PREREGISTRATION.sha256` matches; the live `PREREGISTRATION.md`
hashes to the same value at stamp time. **The protocol is now
immutable.** Any failing pre-registered criterion is reported in
the paper as a null / failure, not amended.

## What was submitted

| Job | Partition | Throttle | Walltime | Array | Purpose |
|---|---|---|---|---|---|
| `p2hpo` 211617 | gpu_a100_8 | %3 | 2-00:00:00 | 0-16 | 17 HPO tasks: 8 backbones × {mse,risk} + ridge |
| `p2ev_hA` 211618 | gpu_h100_4 | %4 | 12:00:00 | 0-999 | EVAL deep chunk A (after HPO) |
| `p2ev_hB` 211619 | gpu_h100_4 | %4 | 12:00:00 | 0-274 (+1000) | EVAL chunk B |
| `p2ev_aA` 211620 | gpu_a100_8 | %3 | 12:00:00 | 0-999 | EVAL mirror A |
| `p2ev_aB` 211621 | gpu_a100_8 | %3 | 12:00:00 | 0-274 (+1000) | EVAL mirror B |
| `p2ev_vA` 211622 | gpu_v100_2 | %2 | 12:00:00 | 0-999 | EVAL mirror A |
| `p2ev_vB` 211623 | gpu_v100_2 | %2 | 12:00:00 | 0-274 (+1000) | EVAL mirror B |
| `p2agg` 211624 | gpu_v100_2 | — | 02:00:00 | — | Aggregator (afterany all eval) |

Dependencies: EVAL `afterany` HPO; AGG `afterany` all EVAL arrays.

## Targets

- 17 / 17 selected configs in
  `/scratch/.../icaif2026/p2out/selected/{model}_{arm}.json`.
- 1275 / 1275 score parquets in
  `/scratch/.../icaif2026/p2out/scores/{model}_{arm}_s{seed}_f{fold}.parquet`.
- 1 `bench/phase2_results.md` with the H1 verdict table from the
  aggregator.

## How the autonomous monitor handles the campaign

A `ScheduleWakeup` cycle (1800 s initially, stretching to 3600 s
once stable) polls:

```
ssh bitshpc squeue -u $USER                  # liveness, failure states
ls /scratch/.../p2out/selected | wc -l       # HPO progress (target 17)
ls /scratch/.../p2out/scores   | wc -l       # EVAL progress (target 1275)
tail -n N /scratch/.../logs/*.err            # any real errors
```

On recurring failures (≥3 tasks with the same error), the monitor:

1. Reads the .err to identify root cause.
2. Common watch-outs and their mitigations:
   - **OOM**: lower `P2_CHUNK` (encoder chunk size) via
     `--export=…,P2_CHUNK=128`. Re-push the changed sbatch line,
     `scancel` the offending pending tasks, re-run
     `phase2_submit.sh` (idempotent).
   - **Quantile-fp16 regression** (already fixed in
     `heads.soft_top_decile` via `.float()`): would resurface if a
     refactor reverts that line. Fix the line, scp, resubmit.
   - **Checkpoint+autocast Optuna trial crash**: pruning
     occasionally interacts with `torch.utils.checkpoint`; mitigated
     by `use_reentrant=False` in `_encode`.
3. After fix: scp the patched file into
   `/scratch/.../icaif2026/engine/`; `scancel` the broken pending
   tasks; re-run `bash scripts/phase2_submit.sh` (skips completed
   cells, resubmits failed ones).

## H=20 follow-on (if H=5 jury PASSes)

If the post-result jury on `bench/phase2_results.md` returns PASS
(or a correctly-disclosed null per k.1), the monitor will launch a
parallel H=20 campaign in a NAMESPACED output dir so the H=5
results are not touched:

- New sbatch / submit: `P2_OUT=$ROOT/p2out_h20`, `P2_H=20`.
- Pre-check: panel has `fwd_ret_20` / `fwd_vol_20` columns
  (the feature builder produces them for all supported H).
- `ensure_frozen()` re-verifies the existing stamp (no re-stamp;
  the doc/sha haven't changed → idempotent).
- HPO and EVAL re-run from scratch for H=20 — the label
  distribution differs from H=5 so neither HPO nor selected
  configs nor scores carry over. This is the honest cost of doing
  H=20.
- Aggregator H=20 writes `bench/phase2_results_h20.md`.

## What the user sees at the end

- `bench/phase2_results.md` (H=5): the primary table —
  per-backbone Sharpe (risk / mse / ridge), Δ-Sharpe vs mse and
  vs ridge, Δrank-IC, p-values, PBO, H1_ACCEPT.
- `bench/phase2_results_h20.md` (if H=20 ran): same schema at H=20.
- The verdict in plain English at the end of each file:
  - "H1 ACCEPTED for ≥1 backbone (see table)", or
  - "H1 NULL for ALL 8 backbones — the pre-registered honest-
    negative outcome (k.1: reported as a failure, NOT amended)".
- A jury report (written by the monitor's spawned subagent) on the
  verdict's soundness, attached to the user's next-day message.

## Hard rules the monitor enforces autonomously

- Never edit `PREREGISTRATION.md`. k.1 binding.
- Never amend the k.6 accept rule. k.1 binding.
- Never re-write a failed criterion to be passable. k.1 binding.
- Never use `$HOME` on the HPC. Standing constraint.
- Never `git push` (environment-blocked classifier). Local commits
  only.
- Never use `scancel` on another user's jobs.

## Why the campaign is multi-day

A single deep cell at d_model=256 on the full train pool with
nested-inner early stop is on the order of 30 min – 4 h on H100,
depending on patience/early-stop trigger. 1275 cells / ~9 effective
concurrent GPUs (4 H100 + 3 A100 + 2 V100 throttle, all heavily
contended on a shared cluster) gives a realistic wallclock of
several days. The first day will mostly be HPO (17 tasks × pruned
64 trials, ~hours per task on A100). EVAL ramps as HPO finishes.

## Files of record (this stage)

- `FREEZE_STAMP` (immutable provenance).
- `bench/phase2_results.md` (forthcoming).
- `p2out/phase2_raw.json` (forthcoming).
- `scripts/phase2_submit.sh` (the operational recovery tool).
- The squeue snapshots in the chat transcript log the live state
  at the time of writing.
