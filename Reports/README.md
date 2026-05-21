# Reports/ — FinSharpe ICAIF-2026 documentation index

Comprehensive, stage-wise documentation of the FinSharpe / ICAIF-2026
project. Each file covers one stage in depth — every decision, every
failure encountered, every fix, every pre-registration amendment, and
the rationale that links them. Designed to be readable independently;
read in numerical order for the full narrative.

| # | File | What it covers |
|---|---|---|
| 00 | [`00_overview.md`](00_overview.md) | One-page snapshot: thesis, scope, target venue, current state, key constraints. |
| 01 | [`01_dataset_and_universe.md`](01_dataset_and_universe.md) | FNSPID dataset profiling, universe construction (Tier-1 870), sensitivity, verification, what was decided and what was discarded. |
| 02 | [`02_sentiment_pipeline.md`](02_sentiment_pipeline.md) | Tier-1 news extraction, FinBERT GPU job on HPC, daily exp-decay sentiment, validation vs FNSPID GPT-3.5 labels, title/body ablation, the date-only de-scope and why. |
| 03 | [`03_features_and_leakage.md`](03_features_and_leakage.md) | Causal feature pipeline, per-stock z fit window, the dedup over-merge failure and its 5.82% cascade fix, leakage QC battery (truncation perturbation, raw recompute). |
| 04 | [`04_preregistration_amendments.md`](04_preregistration_amendments.md) | The 11 amendments a–k verbatim with rationale; what triggered each; the "amend-until-pass" meta-finding that produced k.1. |
| 05 | [`05_juries_chronological.md`](05_juries_chronological.md) | Every adversarial jury round (1 through 7), what each found, what was changed in response, what was preserved un-amended. |
| 06 | [`06_engine_and_models.md`](06_engine_and_models.md) | The 8 in-house backbones (iTransformer, PatchTST, TFT, GCformer, DLinear, LSTM, RNN, CNN), architecture decisions, the strengthening pass to research-grade capacity, verify_models gates. |
| 07 | [`07_objective_and_endpoint.md`](07_objective_and_endpoint.md) | FATAL-1: the original `softmax(mu/tau)` ≠ the §4 endpoint problem; the differentiable soft-top-decile replacement; k.2 surrogate/endpoint dissolution; net-of-cost turnover; tau pinning. |
| 08 | [`08_cpcv_dsr_pbo.md`](08_cpcv_dsr_pbo.md) | Combinatorial Purged CV (6,2)=15 paths, nested-inner HPO split, Deflated Sharpe N=1152 (k.3), CSCV S=10/252 PBO (k.4). |
| 09 | [`09_etth1_anchor.md`](09_etth1_anchor.md) | External validity: the pooled-encoder anchor FAIL (7/8 criterion-iii fail, preserved un-amended), the native-head re-spec PASS, the all-8 derived-gate run (k.5). |
| 10 | [`10_determinism.md`](10_determinism.md) | The seed harness, the real-path 2-run bit-identical proof on the actual Phase-2 training stack (re-jury-6 MAJOR), the AMP-quantile fp16 fix. |
| 11 | [`11_backtest_scorer.md`](11_backtest_scorer.md) | `engine/backtest.py` `score_h1`: the hard endpoint, the turnover burn-in fix, the H1 accept rule (k.6). |
| 12 | [`12_phase2_launcher.md`](12_phase2_launcher.md) | Task 52 — the multi-day SLURM campaign: phase2 driver, freeze module, aggregator, sbatch design, MaxArraySize=1001 workaround, idempotent skip, mirrors across H100+A100+V100. |
| 13 | [`13_freeze_and_campaign.md`](13_freeze_and_campaign.md) | The moment of the PREREG k.1 binding-freeze, what's currently in flight, how H=20 will be launched if H=5 jury PASSes. |
| 14 | [`14_failures_log.md`](14_failures_log.md) | Cross-cutting catalogue of every failure (jury FATALs, MAJORs, code bugs, environment issues) with root cause and resolution. |
| 15 | [`15_limitations_and_threats.md`](15_limitations_and_threats.md) | What this work explicitly does NOT claim. Survivorship-interaction residual bias, single dataset, FNSPID contamination, honor-bound freeze, sentiment de-scope, 8-page constraint. |
| 16 | [`16_decisions_log.md`](16_decisions_log.md) | Chronological decision log: what was chosen at each fork, what the alternative was, why this branch was taken. |
| 17 | [`17_interim_peek_disclosure.md`](17_interim_peek_disclosure.md) | Research-integrity disclosure: interim partial-result peeks during the campaign, what they did and did not touch, the operator's Sharpe question and its refusal, author errors corrected. |

## Conventions used throughout

- **PREREG §N k.x** — section/amendment in `PREREGISTRATION.md` (the
  pre-registered analysis protocol).
- **k.1**, the binding-freeze clause, is **honor-bound, not externally
  enforceable** — the paper discloses this explicitly. The mechanical
  freeze (`engine/freeze.py` + `FREEZE_STAMP` + `PREREGISTRATION.sha256`)
  makes any post-freeze edit tamper-evident, nothing more.
- **C1** = the scope after collapse: the single honest contribution is
  the controlled relative measurement, not an absolute economic claim.
- **H1** = the one pre-registered primary hypothesis (§4 §12 k.6).
- **The "honest negative"** = the project explicitly allows H1 to fail;
  a null result is the informative answer, and k.1 forbids amending
  to manufacture a positive.

## What is currently true (status as of writing)

- The PREREG k.1 binding-freeze fired at `2026-05-20T03:22:49Z` on
  gpunode4 (BITS HPC, A100 80GB). SHA stamped:
  `340e68bcc4f4c327bd39d9fb798c6940e9a63e8c8f1bf349bda9ddd140eee08d`.
- The H=5 Phase-2 campaign is running on the HPC: 17 HPO tasks +
  1275 evaluation tasks + 1 aggregator, mirrored across
  H100/A100/V100 partitions.
- This documentation was written WHILE the campaign was running.
- See `13_freeze_and_campaign.md` for the live operational picture.
