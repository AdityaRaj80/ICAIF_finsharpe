"""CPCV purge/embargo unit test (re-jury-5 FATAL-3).
Proves: correct path count, train/test disjoint, and NO train date within
`purge` rebalance-steps of any test date (since consecutive rebalance dates
are >=H trading rows apart, purge=1 step => >=H-day label embargo).
"""
import math
import numpy as np
import pandas as pd

from dataset import cpcv_folds, nested_inner

# 300 H-spaced rebalance dates (stand-in for the inner-pool sel_dates)
dates = pd.date_range("2013-01-02", periods=300, freq="7D").values \
    .astype("datetime64[ns]")
NG, K, PURGE = 6, 2, 1
fails = []

folds = cpcv_folds(dates, n_groups=NG, k=K, purge=PURGE)
if len(folds) != math.comb(NG, K):
    fails.append(f"path count {len(folds)} != C({NG},{K})={math.comb(NG,K)}")

pos = {d: i for i, d in enumerate(dates)}
min_gap_overall = 10**9
for fi, (tr, te) in enumerate(folds):
    s_tr, s_te = set(tr.tolist()), set(te.tolist())
    if s_tr & s_te:
        fails.append(f"fold{fi}: train/test overlap")
    ti = sorted(pos[d] for d in te)
    tri = sorted(pos[d] for d in tr)
    g = min((abs(a - b) for a in tri for b in ti), default=10**9)
    min_gap_overall = min(min_gap_overall, g)
    if g <= PURGE:                       # purge+embargo violation
        fails.append(f"fold{fi}: train within {PURGE} steps of test (gap={g})")
    # every test date present, train+test+purged partition the timeline
    if len(s_te) == 0 or len(s_tr) == 0:
        fails.append(f"fold{fi}: empty train/test")

# coverage: every date is tested in exactly C(NG-1,K-1) paths
cov = {}
for _, te in folds:
    for d in te:
        cov[d] = cov.get(d, 0) + 1
exp_cov = math.comb(NG - 1, K - 1)
if len(set(cov.values())) != 1 or next(iter(cov.values())) != exp_cov:
    fails.append(f"coverage uneven: {sorted(set(cov.values()))} exp {exp_cov}")

# nested inner split (HPO early-stop): disjoint, val is the tail, gap>=purge
tr0 = folds[0][0]
it, iv = nested_inner(tr0, n_groups=5, val_groups=1, purge=PURGE)
if set(it.tolist()) & set(iv.tolist()):
    fails.append("nested_inner: inner train/val overlap")
if len(it) and len(iv) and pos[iv.min()] - pos[it.max()] < PURGE:
    fails.append("nested_inner: gap < purge")
if len(iv) and iv.min() < tr0.min():
    fails.append("nested_inner: val not from the train fold")

print(f"[cpcv] paths={len(folds)} (C({NG},{K})={math.comb(NG,K)}) "
      f"min train-test step-gap over all folds={min_gap_overall} "
      f"(need >{PURGE})")
print(f"[cpcv] per-date test coverage = {exp_cov} (uniform)")
print(f"[cpcv] nested_inner: inner_train={len(it)} inner_val={len(iv)} "
      f"disjoint+tail OK")
print("RESULT:", "ALL PASS" if not fails else f"FAIL -> {fails}")
