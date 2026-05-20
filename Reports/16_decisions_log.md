# 16 — Decisions log (chronological)

Every fork in the project, what was chosen, what the alternative
was, and why this branch was taken. The "what" sits in the other
docs; this file documents the "why" so a reviewer (or a future
contributor) can audit the reasoning.

## Stage 1: dataset + universe

**D1.1 — Use FNSPID as the base dataset.**
Alternatives considered: CRSP (user lacks BITS access), CMIE Prowess
(US equity coverage thin), Bloomberg / Refinitiv (license cost).
Chosen FNSPID because it is the widely-used public benchmark with
the news pairing that the work needs. Disclosed: contamination
properties (snapshot survivorship, 2021 trough, date-only news).

**D1.2 — Tier-1 (~870) vs Tier-2 vs full ~7693.**
Tier-1 chosen as the backtest universe. Tier-2 kept for ±1-step
sensitivity. Full ~7693 not used because of the long tail of
micro-coverage tickers (a few hundred price rows) and the news
sparsity below the threshold (most modeling would be on missing
data). Disclosed: the gate values + the sensitivity band.

**D1.3 — Common shares only; ETF exclusion list.**
Alternative: include broad ETFs. Rejected because broad ETFs would
dominate the cross-section with low-noise composite series and
flatten the rank-IC; not the population we want to test.

**D1.4 — Exclude 17 repairable-row tickers.**
Alternative: keep them with ad-hoc patches. Rejected because per-
ticker hand-patches introduce a hidden data-quality differential.
Cleaner to exclude and disclose.

## Stage 2: news + sentiment

**D2.1 — FinBERT (ProsusAI) for sentiment scoring.**
Alternatives: BERT-base finetune from scratch, RoBERTa-financial.
Chosen FinBERT for being the de-facto baseline in the literature
and a pre-trained model that runs offline on the HPC GPU. Validated
against FNSPID's GPT-3.5 labels (Task 14); class-agreement in
expected literature range.

**D2.2 — Date-only conservative T+1 alignment (Round-1 fix).**
Alternative: UTC→ET intraday alignment. Rejected because 99.5% of
FNSPID timestamps are date-only — the intraday story is fiction.
T+1 strictly-after prevents boundary leakage; the cost is bounded
signal strength, which is honest.

**D2.3 — De-scope sentiment to descriptive ablation.**
Alternative: claim sentiment as a tested feature. Rejected because
the date-only timestamps + the 2021 trough make any inferential
claim fragile. PREREG §10 explicitly excludes the sentiment Δ from
DSR/PBO/significance. The relative framing cannot then be read as
silently validating sentiment.

## Stage 3: features + leakage

**D3.1 — Per-stock z fit on train (2013-2019) only.**
Alternative: per-stock z fit on full history. Rejected because the
val/test moments would leak into the normalization. Fixed-window
fit is the standard causal choice.

**D3.2 — 504-row lookback.**
Alternative: shorter (256) or longer (1024). 504 chosen as ≈ 2
years — enough to capture annual seasonality and the longer of the
two memory horizons most architectures need; computationally
tractable at d_model=256.

**D3.3 — 65 input features.**
Standard technical + the optional sentiment feature. Larger feature
sets considered (cross-sectional ranks, sector dummies, macro). Kept
to the canonical set because the controlled-contrast principle
demands the same input to all arms; richer features can be a
follow-up.

**D3.4 — Stricter dedup key after over-merge.**
The earlier loose key collapsed real-distinct articles. New key
includes date + source. Cascade fully re-run.

## Stage 4: scope + thesis

**D4.1 — Collapse from C1+C2 to C1-only (re-jury-2).**
Alternative: keep both. Rejected because 8 pages + the disclosed
survivor universe cannot jointly support an economic claim and a
methodology benchmark; the C1 spine is the credible one.

**D4.2 — Reframe to "FNSPID-only methodology re-analysis" (Task 33).**
Alternative: keep dual-dataset pretence. Rejected because CRSP
access is unavailable; a credible second dataset can't be added in
the time budget. Re-jury-4 reinforced: the paired contrast does
not cancel survivorship; the residual is interaction-bias.
Disclosed; absolute metrics moved to appendix only.

**D4.3 — Single H1, H=5, n_eff ≈ 150.**
Alternative: multi-H primary, or pooled-H endpoint. Rejected for
power and multiple-testing reasons; H=5 is the primary, H=20 is a
follow-on robustness arm (now triggered only if H=5 jury PASSes).

## Stage 5: engine + objective

**D5.1 — In-house faithful reimplementations of the 8 backbones.**
Alternative: clone official repos. Rejected because the controlled-
contrast principle requires the same encoder→pooled-vector→head
contract; vendored repos differ on heads, normalization, training
loops. User explicitly authorised: "in-house faithful
reimplementations OK." Impostor concern addressed by the ETT anchor.

**D5.2 — d_model=256 (strengthening pass, Task 51).**
Alternative: lighter d_model=64 or =128. Rejected: user instruction
"don't make the models lightweight, make them strong" + jury concern
that an honest-negative is only credible if the deep models aren't
under-built.

**D5.3 — Differentiable soft-top-decile (FATAL-1 fix).**
Alternative: keep softmax. Rejected: softmax tilt ≠ §4 endpoint;
mathematically false claim. The sigmoid-over-(s-quantile)/(τ·std)
construction has τ→0 limit equal to the hard endpoint (proven in
test_pipeline T4).

**D5.4 — Sigma in NLL only; NOT in allocation.**
Alternative: use sigma to weight the portfolio. Rejected: the
original gate was unused in code, training a head that did nothing
was over-claim. Honest framing: sigma is OPTIONAL calibration; the
allocation uses μ.

**D5.5 — Static τ=0.05; annealing optional (k.2).**
Alternative: anneal τ from 0.10 → 1e-3. Either is k.2-compliant
because validity is on the hard endpoint. Static chosen for
operational simplicity; annealing left available
(`heads.tau_schedule`).

**D5.6 — NET-of-cost turnover term in the training objective
(Task 50).**
Alternative: train with gross Sharpe. Rejected: the trained
objective would NOT match the §4-net eval rule; misalignment
between train and eval. Added; first-rebalance burn-in uncharged
(matches §4 deploy-once).

## Stage 6: statistical machinery

**D6.1 — CPCV(6, 2) for the performance estimator.**
Alternative: simple k-fold. CPCV chosen because it provides
multiple OOF paths + leakage purging in one design (López de
Prado). 6 / 2 chosen as the smallest fold count giving 15 paths.

**D6.2 — CSCV S=10 / 252 paths for PBO (k.4).**
Alternative: PBO via bootstrap. CSCV chosen as the canonical
construction (López de Prado 2016); 252 paths give a low-variance
logit estimate.

**D6.3 — DSR N = 9 × 2 × 64 = 1152 (k.3).**
Alternative: N = (model × arm × seeds × paths × HPO) ≈ 86k.
Rejected: seeds and CPCV paths are variance reduction of a fixed
selected config, not selectable strategies. Counting them inflates
N artificially and makes DSR vacuous. The k.3 number is the
operationally defensible "what the author could have selected."

**D6.4 — N_HPO = 64 with median pruning.**
Alternative: 32 (the prior value, raised in PREREG §9). 64 chosen
because under-tuning large nets would manufacture the negative
(re-jury); pruning keeps wallclock bounded; sampled count remains
64 so DSR N is unchanged.

**D6.5 — HPO selects ONE config per (model, arm), not per-fold.**
Alternative: per-fold HPO. Rejected: would multiply DSR N by 15
(uselessly inflating), and the design wasn't pre-registered as
per-fold HPO. The fixed-selected-config design matches k.3.

**D6.6 — k.6 accept rule fixed BEFORE the run; multiple thresholds
all required.**
Alternative: single-metric accept. Rejected: easier to manufacture
a positive on one metric than on the conjunction. Conjunction of
ΔDSR ≥ 0.20, Δrank-IC ≥ 0.01, PBO ≤ 0.5, p < 0.05 vs BOTH
comparators is the conservative rule.

## Stage 7: pre-registration discipline

**D7.1 — Pointer-only PREREG (amendment c).**
Alternative: keep measured numbers in PREREGISTRATION.md.
Rejected: SHA was anchoring numbers, not just a-priori text →
goalpost-shifting risk. All measured numbers stripped; doc anchors
only the pre-registered text/structure.

**D7.2 — Preserve the failed (iii) criterion (amendment j).**
Alternative: relax (iii) to make all-8 PASS. Rejected: that is the
amend-until-pass anti-pattern. Failed numbers stand; the valid
fidelity test is re-specified separately (native head; 9a-NATIVE)
without overwriting the prior failure.

**D7.3 — k.5: gates = 1.5 × worst published.**
Alternative: gates eyeballed. Rejected: must be a rule, not a
number; multiplier fixed BEFORE any run; citations attached.

**D7.4 — k.1 BINDING-FREEZE CLAUSE (the final amendment).**
Alternative: allow further amendments. Rejected: re-jury-6
finding that no failed criterion in a–j had ever survived
un-amended makes an honest-negative non-falsifiable. k.1 closes
the chain; failures after the first Phase-2 train are reported as
failures, never amended. Honor-bound, disclosed.

## Stage 8: launcher + HPC

**D8.1 — Use A100 too (mid-session override).**
The earlier constraint was H100/H200 only ("A100s are slower").
User later said "all three GPUs — H100, H200 and A100s, whichever
is free we will utilize." Cluster has no H200 → mirrors on
H100 + A100 + V100. Idempotent skip makes mirrors race-safe.

**D8.2 — Idempotent skip + DEFER.**
Alternative: hard-fail on missing dependencies. Rejected: a single
HPO straggler would block all 1275 eval cells under afterok.
Idempotent + DEFER means re-running the submit script is the
recovery tool; failed cells retry, completed ones skip.

**D8.3 — MaxArraySize=1001 → IDX_OFFSET chunking.**
Alternative: 5 arrays of 255 tasks each. Rejected: more SLURM
overhead, more dependency edges. Two chunks per partition (0-999,
0-274 +offset 1000) is the minimum-overhead design.

**D8.4 — All I/O on /scratch, never $HOME.**
Standing operational constraint (shared HPC account, $HOME quota
small). Reaffirmed throughout. All caches, pylibs, panels, scores,
logs land under `/scratch/goyalpoonam/finsharpe/icaif2026/`.

**D8.5 — Local commits only; never `git push` from this environment.**
The data-exfiltration classifier blocks `git push` to the GitHub
remote. scp / ssh to the user's own HPC account is legitimate
research-data staging and not blocked. User pushes manually when
ready.

## Stage 9: live operations (autonomous loop)

**D9.1 — Monitor every 1800s initially, stretching to 3600s.**
Per ScheduleWakeup guidance: cache TTL 300s; bigger intervals
amortise the cache miss. 1800s for early stability; 3600s once
green. No 300s polling — it's the worst-of-both.

**D9.2 — Spawn a SUBAGENT for the post-result jury.**
Alternative: do the jury inline. Rejected: a subagent has no
context bias from this conversation → more independent. The jury
operates on `bench/phase2_results.md` alone.

**D9.3 — H=20 only after H=5 PASSes; H=20 in namespaced
`p2out_h20/` so H=5 is not touched.**
The user's explicit sequencing.
