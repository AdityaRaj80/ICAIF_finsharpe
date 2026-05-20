# 04 — Pre-registration and the 11 amendments (a–k)

`PREREGISTRATION.md` is the pre-registered analysis protocol — it
pins, BEFORE any model is trained, every choice that could otherwise
be tuned to manufacture a positive result. It is the spine of the
paper's credibility.

The doc was amended 11 times (a through k). This file lists every
amendment chronologically with the trigger that caused it and the
fix. `04` is intentionally the source-of-truth narrative; `05` lists
the juries (the triggers) chronologically.

## Why amend at all?

Pre-execution refinement is legitimate. The protocol responds to
adversarial juries BEFORE training has begun — that is exactly what
pre-registration is for. What is NOT legitimate is amending AFTER
results are in, to dress a null up as a positive. The danger is the
**"amend-until-pass" pattern**: every failed criterion gets amended,
which makes an honest-negative non-falsifiable. The amendment chain
a–j was scrutinised for this pattern (re-jury-6 was decisive); the
finding was concerning enough that **amendment k.1** was added to
permanently close the chain. See `k` below.

## The amendments

### a — 2026-05-19, initial freeze

The "3-leg" initial freeze: thresholds, hyperparameters, endpoint,
seeds. Set the seven primary numbers (FRAC=0.10, COST=0.001,
COMET, etc.), the 5-seed plan, and the C2 scope.

### b — 2026-05-19, re-jury-2 response

- Scope **collapsed to C1**. The earlier C1+C2 scope tried to do
  both a methodology benchmark and an economic claim; this is what
  the 8-page constraint and the survivor-universe disclosed
  limitation could not jointly support.
- Doc renaming + tightening.
- §3 / §4 overlapping-return inference fixed (originally allowed
  H-overlapping return windows in the inference series; corrected
  to strictly non-overlapping H-spaced rebalances → n_eff goes from
  inflated to the honest ~150 at H=5).
- §3 / §7 early-stop reconciled (the early-stop rule referenced was
  not the one actually used in code; aligned).
- Sentiment moved to conservative date-only.

### c — 2026-05-19, re-jury-3 response (pointer-only doc)

- **All measured numbers removed** from `PREREGISTRATION.md`. The
  doc became pointer-only — the SHA anchors *a-priori text only*,
  not data. Numbers live in code and reports; the doc references
  them.
- H1 binding horizon explicitly restricted to **H=5** with a
  pre-registered MDE/power statement (Lo 2002 SE; α=0.05, power 0.80
  → MDE on the Sharpe-difference ≈ 0.46/√1).

### d — re-jury-3 follow-up: PBO and N

- DSR multiple-testing count tightened.
- PBO via CSCV explicitly noted as the primary overfitting control
  (robust to misspecified N).

### e — sentiment de-scope

- The "sentiment on/off Δ is reported DESCRIPTIVELY ONLY and is
  explicitly EXCLUDED from the DSR/PBO/significance machinery
  applied to H1" wording added in §10.
- Title/body, intraday, label-quality were *deliberately dropped*.
  The relative framing is forbidden from being read as silently
  validating sentiment.

### f — universe sensitivity wording

- Sensitivity band ±1-economic-step disclosed as a band, not as a
  robustness claim.
- 17 repairable-row tickers excluded explicitly.

### g — "frozen" framing corrected

- The protocol is **not yet frozen during development**; it becomes
  immutable at the first Phase-2 training step (re-jury-4 fix).
  Until then, amendments are legitimate pre-execution refinement.

### h — sentiment validation de-scope

- The FinBERT vs GPT-3.5 validation result is descriptive only.
- The title/body ablation is descriptive only.
- The intraday-handling investigation is RETIRED, not concluded.

### i — first ETT anchor design

- Pre-registered the external-validity anchor on standard public
  ETTh1, with the criterion list (i)-(iv) fixed BEFORE seeing any
  number.

### j — ETT anchor re-spec (after the run)

- 9a-OUTCOME preserved: criterion (iii) `< 0.80` FAILED for 7/8
  models in the pooled-encoder configuration. **Not amended; the
  failed numbers stand.**
- 9a-NATIVE added: a *separate* fidelity criterion using each
  fidelity-critical model's native forecasting head + RevIN-style
  normalization, the configuration in which the published ETTh1
  numbers were actually obtained. PASS criteria for the trio
  {DLinear, PatchTST, iTransformer} fixed BEFORE its run.
- The two are clearly distinguished. The pooled-head failure is
  preserved un-amended; the native-head test is what the paper cites
  for fidelity.

### k — 2026-05-20, FINAL & BINDING

The decisive meta-finding from re-jury-6: across a–j, no failed
criterion ever survived un-amended (with the partial exception of
9a-OUTCOME, which was preserved alongside a re-spec). This pattern
makes an honest-negative non-falsifiable.

**k.1 — BINDING-FREEZE CLAUSE.** The protocol freezes at the first
Phase-2 training step; the SHA is re-stamped then and the document is
not edited afterward. **Any pre-registered criterion that fails after
this amendment is reported in the paper AS A FAILURE / null result —
it is NOT amended, re-specified, or scope-reduced.** No amendment l.
The a–j iteration was legitimate disclosed pre-execution refinement;
it is now closed.

**k.2 — Surrogate ≠ endpoint (dissolves re-jury-6 FATAL).** The
differentiable soft-top-decile is ONLY the *training surrogate*. H1
and the §4 backtest are evaluated by applying the HARD long-only
top-decile equal-weight rule to the trained model's output scores
(`engine/backtest.py` `score_h1` → `hard_top_decile_returns`). No
claim that the surrogate equals the endpoint; whether optimising the
surrogate moves the endpoint is precisely what H1 asks. τ MAY be
annealed (`heads.tau_schedule`) but validity does NOT depend on it;
static τ=0.05 is fully compliant.

**k.3 — DSR multiple-testing N.** Deflated-Sharpe
`N = (n_models × n_arms) × N_HPO = (9 × 2) × 64 = 1152`.
Seeds and CPCV paths are variance reduction of a *fixed* selected
config, not selectable strategies, so excluded from N. PBO (CSCV) is
the primary overfitting control; DSR with N=1152 is reported
conservatively as secondary.

**k.4 — PBO via canonical CSCV.** PBO computed by Combinatorial
Symmetric CV with `S = 10` groups → `C(10,5) = 252` train/test
recombinations (López de Prado 2016), distinct from the CPCV(6,2)
used for the performance point estimate.

**k.5 — Mechanically-derived anchor thresholds, all 8 backbones.** No
eyeballed numbers. For each architecture, native-head ETTh1
(96→96, MSE) PASS gate = **1.5 × the worst published ETTh1-96 MSE for
that class** (multiplier fixed a-priori). Frozen worst-published
values are cited with traceable sources:
DLinear .40 [Zeng et al., AAAI 2023] → gate .60;
PatchTST .41 [Nie et al., ICLR 2023] → .62;
iTransformer .39 [Liu et al., ICLR 2024] → .59;
GCformer/transformer-family .45 [Autoformer, NeurIPS 2021] → .68;
TFT .60 [Lim et al. 2021, conservative upper] → .90;
TCN .55 [SCINet/LTSF conv-baseline] → .83;
LSTM .70 / RNN .70 [classical-RNN rows, conservative] → 1.05 each.
Anchor capacity is deliberately d_model=128 (ETT-standard scale);
the FinSharpe campaign capacity d_model=256 is intentionally
decoupled.

**k.6 — Pre-registered primary decision rule (no post-hoc latitude).**
H1 is accepted iff, for the H1 backbone, paired
ΔDSR ≥ 0.20 AND Δrank-IC ≥ 0.01 AND PBO ≤ 0.5 AND p<0.05, at H=5,
vs both same-backbone-MSE and Ridge. Any other outcome ⇒ H1
rejected/null, reported as such. The primary results table schema is
frozen in `PAPER_PLAN.md`.

## The "Protocol-iteration disclosure" section in the paper

The paper WILL contain a "Protocol iteration history" subsection
reproducing the amendments a–k **verbatim**, stating plainly:
(i) the protocol was adversarially iterated 11× pre-execution;
(ii) one pre-registered criterion that FAILED is preserved on the
record (§9a-OUTCOME); (iii) the binding-freeze (k.1) is honor-bound
and disclosed, NOT externally enforceable. This converts the only
un-priced structural risk (the amend-until-pass perception) into a
disclosed, reviewer-judgeable limitation rather than a hidden one.

## SHA stamp

`PREREGISTRATION.sha256`:
```
340e68bcc4f4c327bd39d9fb798c6940e9a63e8c8f1bf349bda9ddd140eee08d  PREREGISTRATION.md
# FINAL (re-jury-7 consistency pass: NO criterion changed; immutable at first Phase-2 train; no amendment l)
```

The matching `FREEZE_STAMP` was written at the first Phase-2 task on
2026-05-20T03:22:49Z (`13_freeze_and_campaign.md`).
