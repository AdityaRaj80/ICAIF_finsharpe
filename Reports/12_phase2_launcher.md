# 12 — Phase-2 launcher (Task 52)

The launcher turns the engine into a multi-day SLURM campaign on
the BITS HPC. Its design constraints:

- **Never** write to `$HOME`. All I/O on
  `/scratch/goyalpoonam/finsharpe/icaif2026/`. `optuna` installed
  into `/scratch/.../pylibs` via `pip --target`.
- Shared HPC account — throttle array concurrency to be a good
  cluster citizen.
- Restart-safe across many days of cluster churn.
- Idempotent: re-running the submit script must double as the
  recovery tool (no double-work, no clobbering completed cells).
- The first task fires PREREG k.1 binding-freeze; the freeze
  must be race-safe across concurrent array tasks.

## Files (this stage)

```
engine/freeze.py             # the binding-freeze
engine/phase2.py             # the driver (hpo / cell / ridge / dispatch)
engine/phase2_aggregate.py   # the H1 verdict aggregator
scripts/phase2.sbatch        # the generic SLURM worker
scripts/phase2_stage.sh      # login-node staging (untar + pip optuna)
scripts/phase2_submit.sh     # the orchestrator (HPO -> EVAL -> AGG)
FREEZE_STAMP                 # the immutable provenance, stamped at first task
PREREGISTRATION.sha256       # the SHA the freeze verifies against
```

## `engine/freeze.py` — the k.1 binding-freeze

```python
def ensure_frozen(job="phase2"):
    live = sha256_of(DOC)
    rec  = recorded_sha()
    if live != rec:
        sys.exit("FREEZE-ABORT: doc edited, integrity broken (k.1)")
    if exists(STAMP):
        verify(rec in head_line(STAMP))         # post-freeze edit ?
        return rec
    try:                                        # race-safe across SLURM
        fd = os.open(STAMP, O_CREAT|O_EXCL|O_WRONLY, 0o444)
        write(fd, f"{rec}  frozen_utc={…}  host={…}  job={job}\n")
    except FileExistsError:
        verify(rec in head_line(STAMP))
    return rec
```

Two paths (DOC, SHAFILE, STAMP) are env-overridable
(`P2_FREEZE_DOC`, `P2_FREEZE_SHA`, `P2_FREEZE_STAMP`) — the local
smoke test points them at a throwaway tmp dir, so the smoke can
exercise the freeze logic without prematurely locking the real
protocol.

## `engine/phase2.py` — the driver

Four sub-commands:

- **`hpo --model M --arm {mse,risk}`** — 64-trial Optuna TPE with
  median pruning; writes `selected/{M}_{arm}.json`. ridge is also
  here (closed-form regression; tunes only the L2 α).
- **`cell --model M --arm A --seed S --fold F`** — train selected
  config on CPCV(6,2) path F's train; predict path F's test; write
  `scores/{M}_{A}_s{S}_f{F}.parquet`.
- **`ridge --seed S --fold F`** — closed-form Ridge per (seed, fold).
- **`dispatch --phase {hpo,eval} --idx N`** — single source of truth
  mapping `SLURM_ARRAY_TASK_ID` → work unit:
  - HPO: `idx 0..15` deep `(BACKBONES[idx//2], ARMS[idx%2])`;
    `idx 16` → ridge.
  - EVAL: `idx 0..1199` deep
    `(BACKBONES[idx // 30], ARMS[(idx % 30) // 15],
      seed=(idx % 15) // ?, fold=idx % ?)` — the actual derivation
    is `fold = idx%15; t=idx//15; seed=t%5; t=t//5; arm=ARMS[t%2];
    model=BACKBONES[t//2]`. `idx 1200..1274` → ridge.

### Why the work grid is 17 + 1275 tasks

- HPO: 8 backbones × 2 arms = 16, plus Ridge = **17**.
- EVAL deep: 8 × 2 × 5 seeds × 15 CPCV paths = **1200**.
- EVAL ridge: 5 × 15 = **75**.
- Total array tasks: 17 + 1275 + 1 aggregator = 1293 SLURM jobs.

### Memory-safe encoder forward

The cross-section per date is up to ~870 stocks × `MIN_DATES=16`
dates = ~14000 sequences of `504 × 65` floats. At `d_model=256` a
naïve full-batch encoder forward would OOM even on 80 GB. The
driver does **gradient-checkpointed chunked encoding** via
`torch.utils.checkpoint.checkpoint(enc, x_chunk,
use_reentrant=False)` per `CHUNK=256` (env-tunable `P2_CHUNK`):
activation memory drops to `O(CHUNK)`, the backward recomputes
each chunk's forward, and the gradient stays exact and
deterministic.

### HPO search space (optimizer-only)

Architecture and the pinned PREREG-§9 constants are NOT searched.
The space (per trial):
- `lr ∈ loguniform[1e-4, 3e-3]`,
- `wd ∈ loguniform[1e-7, 1e-3]`,
- `dropout ∈ {0.0, 0.1, 0.2}`,
- `grad_clip ∈ {0.5, 1.0, 5.0}`,
- `patience ∈ {4, 6, 10}`.

Ridge: only `alpha ∈ loguniform[1e-3, 1e3]`.

Optuna TPE sampler seeded at 0; **MedianPruner** with
`n_startup_trials=8, n_warmup_steps=5`. Pruning early-kills
hopeless trials but the *sampled* trial count remains 64, so the
DSR `N = 9 × 2 × 64 = 1152` is unchanged (k.3); this is disclosed
in the code.

### Idempotent skip + DEFER guard

- `run_cell` first checks `_done(model, arm, seed, fold)` — if the
  output parquet already exists, **skip** (print and return 0).
- If the cell's selected-config JSON is missing (HPO not done yet),
  **DEFER** (print and return 0). This lets the eval array
  `--dependency=afterany` on HPO be safe: if HPO has a straggler,
  eval cells that need it just no-op until the operator re-runs
  `phase2_submit.sh`.
- Same idempotent/DEFER logic in `run_ridge` and `run_hpo`.

Net effect: re-running `bash scripts/phase2_submit.sh` is the
recovery tool. Completed cells skip. Newly-runnable cells pick up.

## `engine/phase2_aggregate.py`

When all cells finish, the aggregator:

- Loads every `scores/{model}_{arm}_s{seed}_f{fold}.parquet`.
- Averages each `(date_id, sym_id)`'s `score` over **seeds AND CPCV
  paths where the cell is out-of-sample** — the k.3 variance-
  reduction average for a fixed selected config.
- Restricts the H1 scoring series to the **test-split rebalance
  dates** (PREREG §1: H=5 backtest is the test period, target
  `n_eff ≈ 150`). The local smoke independently reproduced
  `test-split rebalance points = 149` — matching the §1 target
  exactly.
- Calls `backtest.score_h1(risk_oof, mse_oof, ridge_oof, y, did,
  sym_id)` per backbone.
- Writes `bench/phase2_results.md` (primary table + per-backbone
  H1_ACCEPT + overall verdict) and `p2out/phase2_raw.json`.

Per-seed Sharpe dispersion is computed (descriptive appendix only)
and never enters DSR/PBO/k.6.

## `scripts/phase2.sbatch` — the generic worker

Sets the env that every task needs:

```bash
export P2_PANEL=$ROOT/panel/features.parquet
export P2_SYMS=$ROOT/universe/tier1.txt
export P2_OUT=$ROOT/p2out
export P2_H=${P2_H:-5}
export PYTHONPATH=$BASE/pylibs:$ROOT/engine
export PYTHONNOUSERSITE=1
export HF_HOME=$BASE/.hf TORCH_HOME=$BASE/.torch
export XDG_CACHE_HOME=$BASE/.cache TMPDIR=$BASE/tmp
export CUBLAS_WORKSPACE_CONFIG=:4096:8
IDX=$((SLURM_ARRAY_TASK_ID + ${IDX_OFFSET:-0}))
srun "$PY" engine/phase2.py dispatch --phase "$PHASE" --idx "$IDX"
```

Generic — partition / qos / array range / throttle / time are passed
on the `sbatch` command line so the SAME script serves H100, A100,
and V100.

## `scripts/phase2_submit.sh` — the orchestrator

```
HPO  --partition=gpu_a100_8 --qos=qos_gpu_a100 --array=0-16%3
     --time=2-00:00:00
EVAL --partition=gpu_h100_4              --array=0-999%4 [+IDX_OFFSET=1000 -> 0-274%4]
EVAL --partition=gpu_a100_8 --qos=...    --array=0-999%3 [+IDX_OFFSET=1000 -> 0-274%3]
EVAL --partition=gpu_v100_2              --array=0-999%2 [+IDX_OFFSET=1000 -> 0-274%2]
       --dependency=afterany:<hpo>
AGG  --partition=gpu_v100_2 --time=02:00:00
       --dependency=afterany:<all eval arrays>
```

### MaxArraySize = 1001 — the index workaround

SLURM has `MaxArraySize = 1001` on this cluster. That is the **max
INDEX** allowed in a single array (not just a count). Our 1275 eval
tasks therefore split into two chunks per partition:
- chunk A: `--array=0-999%N IDX_OFFSET=0`,
- chunk B: `--array=0-274%N IDX_OFFSET=1000`.

`phase2.sbatch` computes `IDX = SLURM_ARRAY_TASK_ID + IDX_OFFSET`
before calling `dispatch`.

### Mirrors across H100 + A100 + V100 = "whichever is free"

Each chunk is submitted to ALL THREE partitions concurrently. The
idempotent skip makes the mirrors race-safe: whichever GPU frees
first runs a given task; later attempts skip. The user's instruction
was explicit: "all three GPUs — H100, H200 and A100s, whichever is
free we will utilize" (and this cluster has no H200 partition — V100
fills that slot).

Throttle: `%4` (H100), `%3` (A100), `%2` (V100). Modest, to share
the cluster fairly.

### HPO is NOT mirrored

HPO runs only on A100 (5-day partition walltime). Mirroring HPO
would risk two concurrent HPO tasks for the same (model, arm) both
running 64 trials and the second wasting compute. Single partition
prevents duplicate HPO work.

## Cluster facts (confirmed at submission)

- `gpu_h100_4`: 2 nodes (gpunode5, gpunode6) × 4 H100-80GB =
  8 GPUs total. MaxTime 2 days. `QoS=N/A AllowQos=ALL`.
- `gpu_a100_8`: 1 node (gpunode4) × 8 A100-80GB. MaxTime 5 days.
  `QoS=qos_gpu_a100`.
- `gpu_v100_*`: 3 V100 GPUs across gpunode1-3 (fallback).
- `MaxArraySize=1001`, `MaxJobCount=10000`.

## Local micro-smoke (before launch)

`P2_SMOKE=1` shrinks d_model→32, MAX_EPOCH→2, N_HPO→4, universe→12
symbols, MIN_DATES→4. The smoke:
- Stamps the freeze in a `tmp/STAMP` (real PREREG untouched).
- Runs HPO (ridge + deep risk + deep mse) and writes selected JSONs.
- Runs `cell` (risk and mse) and `ridge` and writes score parquets.
- Re-running shows idempotent skip ("skip ridge s0 f0 (exists)").
- Runs `dispatch --phase eval --idx N` to verify the index map
  (0→`itransformer mse s0 f0`, 1200→`ridge s0 f0`, etc.).
- Runs the aggregator to verify it routes through `score_h1` (the
  micro data is too thin to satisfy the k.6 rule — that is correct
  for a smoke; the routing path is what is being tested).

All of this PASSED before the real campaign was submitted.

## Files of record (this stage)

- `engine/freeze.py`, `engine/phase2.py`,
  `engine/phase2_aggregate.py`.
- `scripts/phase2_stage.sh`, `scripts/phase2.sbatch`,
  `scripts/phase2_submit.sh`.
- `FREEZE_STAMP` (the immutable provenance, see
  `13_freeze_and_campaign.md`).
