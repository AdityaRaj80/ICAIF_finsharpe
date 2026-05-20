# 15 — Limitations and threats to validity

The work makes a narrow, controlled claim. This file is what the
paper's "Limitations" section will pull from. Every item here is
disclosed up front (in the abstract and in PAPER_PLAN), not buried.

## L1. FNSPID dataset contamination

### L1.a. Selection survivorship (universe)

The 870-name Tier-1 universe was **full-sample selected**
(completeness / liquidity / news gates evaluated over the entire
window). Tickers that failed during the window are not in the
universe. The universe over-represents winners.

**Mitigation (partial, NOT cancellation).** The paired H1 contrast
trades both arms on the same universe → many forms of unconditional
survivorship cancel. The residual is **survivorship-interaction
bias**: two different objectives pick different sub-portfolios with
different exposure to the missing delisted mass → the paired
difference can be biased in a self-favoring direction. This is
disclosed as a primary limitation (PAPER_PLAN: "Survivorship:
disclosed, NOT assumed away").

**Sign-bound.** The thin internal early-stop / delisted cohort
provides a small signed bound on the residual; the bound is NOT a
cure. Absolute / level metrics are appendix-only ("levels NOT
investable; completeness only").

### L1.b. No clean delisted cohort

FNSPID is a 2020 snapshot; tickers delisted before then are
under-represented; tickers delisted after vanish. There is no
clean delisted cohort to construct a proper survivor-bias-free
out-of-sample benchmark.

### L1.c. 2021 news trough

The news corpus thins materially in 2021, inflating apparent
stability of sentiment-driven features in that year. The sentiment
feature is de-scoped descriptively only (excluded from
DSR/PBO/significance, PREREG §10) so this does not pollute H1.

### L1.d. 99.5% date-only timestamps

The vast majority of news rows have no intraday timestamp. Any
"publish-minute aware" alignment would be fiction. Sentiment is
aligned conservatively T+1 strictly-after, which prevents leakage
but bounds the signal strength achievable from the date-only data.

## L2. Single dataset → limited external validity

C1 is one dataset. The paper does not claim generalisation to
markets, asset classes, or news pipelines outside FNSPID. The user
considered adding a second dataset (CRSP, CMIE Prowess, others);
none were accessible at BITS or fit the 8-page constraint without a
large engineering tail. The decision to stay FNSPID-only is
disclosed; the framing is "rigorous methodology re-analysis of one
widely-used but contaminated public benchmark," not a generalisation
claim.

## L3. In-house faithful reimplementations vs vendored repos

The 8 backbones are faithful in-house reimplementations, not
checkouts of official authors' repos. This is necessary for the
controlled-contrast principle (the same encoder→pooled-vector→head
contract across all 8) but raises an impostor-implementation
concern. The ETT anchor (`09_etth1_anchor.md`) is designed to
detect that:

- Pooled-encoder criterion (iii) failed 7/8 (preserved un-amended;
  honest-finding: pooled encoder is a multi-step forecasting
  bottleneck, not a model bug).
- Native-head 9a-NATIVE re-spec PASS for the trio
  {DLinear, PatchTST, iTransformer}.
- All-8 mechanically-derived k.5 gates PASS (.389-.574 vs gates
  .59-1.05).

The ETT anchor proves architecture mechanism fidelity, NOT SOTA
reproduction. The paper does not claim SOTA reproduction of any
backbone.

## L4. Honor-bound (not externally enforceable) binding-freeze

k.1 forbids amending the protocol after the first Phase-2 train.
The mechanism (`engine/freeze.py` + `FREEZE_STAMP` +
`PREREGISTRATION.sha256` + the live re-verify on every task) makes
any post-freeze edit tamper-evident. But there is no third-party
notary; the author could in principle still edit. The paper
discloses this explicitly. What the mechanism *does* give a
reviewer:

- An immutable provenance record (`FREEZE_STAMP`: utc, sha, host,
  jobid).
- A verifier (`engine/freeze.py`) the reviewer can run to confirm
  the live doc still hashes to the recorded SHA.
- The full a–j amendment chain reproduced verbatim in the paper
  ("Protocol iteration history") so the reviewer can price the
  amend-until-pass risk themselves.

This converts an un-priced structural risk into a disclosed,
reviewer-judgeable limitation. It does not eliminate it.

## L5. Sentiment de-scoped

Sentiment is one optional feature, descriptively only. The
sentiment-on vs sentiment-off Δ is **explicitly excluded** from
DSR/PBO/significance (PREREG §10). The paper does not claim
sentiment is or is not a useful feature. Title/body, intraday, and
label-quality investigations are reported and retired; the
relative framing cannot be read as silently validating sentiment.

## L6. Power: H=5 MDE ≈ 0.46 (annualised Sharpe units)

Under Lo 2002 SE, α=0.05, power 0.80, `n_eff ≈ 150` (3 years of
non-overlapping H=5 periods), the minimum detectable effect on the
Sharpe-difference is ≈ 0.46 in annualized Sharpe units. The k.6
accept rule's `ΔDSR ≥ 0.20` is detectable only if the underlying
Sharpe gap is at least the MDE. This is disclosed up front: a tiny
true effect would null-out under this test, by design.

## L7. Multi-mirror determinism caveat

Each (seed) is bit-reproducible *per hardware*. The H100 / A100 /
V100 mirrors of the same array cell run on different SKUs, so the
score parquet for a given (model, arm, seed, fold) cell may differ
in the last few ULPs depending on which mirror got there first.
Idempotent skip means whichever finishes first wins. For the H1
aggregator (averaging over folds + seeds, then routing through
score_h1's annualised arithmetic), this is far below the noise
floor. Disclosed.

## L8. 8-page constraint

ICAIF '26 main track is 8 pages sigconf, no supplementary. Many of
the items above can only get a sentence in the paper; the depth
lives in `Reports/` and the public code repo. This compresses the
ability to defend nuances — the paper has to be very crisp about
"this is what we claim and what we don't."

## L9. No external sponsor / no independent replication run

The full multi-day campaign is run on the author's HPC time. No
independent replication is performed before submission. The
campaign is restart-safe and reproducible from the SHA-frozen
PREREG + the engine code + the FNSPID raw — a reviewer with HPC
access could in principle re-run it.

## L10. PBO / DSR are approximations

CSCV-PBO and Bailey-LdP DSR are well-established but
approximations. PBO is the primary overfitting control because it
is robust to N misspecification, but its 252-recombination logit
sample is still finite. DSR with N=1152 is conservative; the
literature has debated whether N should be the number of trials
the *author* searched or the number any author *could* have
searched. We chose the former (k.3) as the operationally
defensible number tied to our actual HPO budget.

## L11. Honest-negative is the most likely outcome

The paper is upfront that the most likely H1 outcome is a NULL
result (deep objectives do not beat Ridge under leakage /
deflation / PBO control). The carrying contribution is precisely
the honest measurement, not the loss. This is unusual for an
ICAIF submission and explicitly framed as the contribution.

## What this paper does NOT claim

- Survivorship-bias-free economic performance.
- Investability of the absolute numbers in the appendix.
- Generalisation outside FNSPID / US equities / the modeled window.
- SOTA reproduction of any individual backbone.
- That the differentiable surrogate equals the hard endpoint
  (k.2 is explicit: they are different objects).
- That sentiment is a useful feature.
- That the binding-freeze is externally enforceable (it is
  honor-bound, disclosed).
- A guarantee that the H1 verdict is positive.
