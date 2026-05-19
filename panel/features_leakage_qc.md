# Feature Pipeline — Independent Leakage QC

_2026-05-19. Re-implemented representative features independently; decisive test = truncation/perturbation causality._

**VERDICT: PASS** — features 2,844,044 rows x 79 cols, 870 symbols.

## Gates

- PASS — G1 perturbation causality (9 feature families incl SENT_MA): future rows cannot change feature[t]
- PASS — G1c XS_ raw-recompute truncation test (builder-independent, >=100 date-col checks)
- PASS — G2a stored norm stats == TRAIN-only moments
- PASS — G2b stored norm stats != all-rows moments (no val/test leak)
- PASS — G2c train-rows normalized ~ mean0/std1 per stock
- PASS — G3 fwd_ret_H == forward log-return (recomputed)
- PASS — G4 sentiment join+norm exact on invertible region (recon err<1e-4)
- PASS — G5 split tags monotone & non-overlapping

## G1 causality (decisive)
- checked 246,012 feature-cells across 12 symbols x 9 families (RET_20, MA_20, KMID, CORR_20, RSV_20, VMA_20, VSTD_20, MOM_20, SENT_MA_20) at a 70%-truncation cut; mismatches when future rows deleted = **0** (must be 0). G1c: XS_ same-date checks 120, violations 0. Proves every feature at t uses only data <= t.

## G2 normalization
- stored (mu,sigma) == train-only moments: True; provably excludes val/test: True; train-rows z ~ N(0,1): True. Per-stock normalization fit on TRAIN ROWS ONLY.

## G3 targets
- fwd_ret_5 vs recomputed forward log-return mismatches = 0 (must be 0); labels are forward by design.

## G4 sentiment join+normalization fidelity
- reconstruction max|SENT*sigma+mu - sent_decay| on un-clipped region = **1.19e-07** (<1e-4 required) -> join + per-stock train-fit normalization are exact. Intentionally winsorized (|z|>=8) tail = 0.00% of joined rows (documented transform, not a defect). Pooled Pearson 0.9756 is <1 BY DESIGN (per-stock z) — context only.

## G5 splits
- per-symbol date-monotone: True; train<val<test non-overlap: True.

## Note
- CPCV purge/embargo = H td is applied at MODEL-TRAIN time per PREREGISTRATION.md §3, not here; this panel only tags calendar splits. Re-run on any feature/sentiment change.