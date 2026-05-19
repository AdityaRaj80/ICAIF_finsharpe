# Title-vs-Body FinBERT Ablation (Jury5#3)

_n=6000 body-bearing nasdaq-source Tier-1 articles; local GPU. ADVISORY:
C3 is DROPPED (PREREG §10) — sentiment is a de-scoped optional feature;
this ablation rides on no claim._

- title/body sign agreement: **0.690**
- title/body score Spearman: **0.483**
- TRAIN fwd-5d return IC: title **+0.0292** vs body **-0.0526** (n_title=329, n_body=329)

## Verdict

- Title/body diverge (agreement 0.690), but C3 is DROPPED — sentiment is only a de-scoped optional feature; this is advisory and rides on no claim.
- Title chosen a-priori (51% of FNSPID bodies absent, FNSPID-comparable), not outcome-selected. Advisory only (PREREG §10).

_elapsed 144s_