"""Unit tests on a REAL small panel subset (re-jury-5 hardened).

T1 loader contract; T2 soft-top-decile portfolio parity vs independent
numpy; T3 ENFORCED (not waived) per-symbol trading-row non-overlap >= H;
T4 operator-limit: tau->0 => uniform over exactly the top-decile names
(== PREREG §4 hard endpoint).
"""
import numpy as np
import pandas as pd
import torch

from dataset import DateGroupedLoader, SEQ_LEN
from heads import portfolio_returns, soft_top_decile, FRAC

H, TAU, MIN_D = 5, 0.05, 8
syms = pd.read_csv(r"D:\Study\FinSharpe\universe\tier1.txt",
                   header=None)[0].astype(str).str.strip().tolist()[:40]
fails = []

print("[t] building loader (test split, H=5, 40 symbols)...", flush=True)
L = DateGroupedLoader("test", H, symbols=syms, min_dates=MIN_D)
sd = L.sel_dates
b = next(iter(L.batches()))
nd = int(b["date_id"].unique().numel())
per = pd.Series(b["date_id"].numpy()).value_counts()

# T1 loader contract
if nd != MIN_D:
    fails.append(f"T1a: batch {nd} dates != {MIN_D}")
if per.min() < 5:
    fails.append(f"T1b: a date has {per.min()} stocks (<5)")
if len(np.unique(sd)) != len(sd):
    fails.append("T1c: duplicate rebalance dates")

# T2 soft-top-decile portfolio parity vs independent numpy
score = b["y_ret"].clone()
port_t, uniq = portfolio_returns(score, b["y_ret"], b["date_id"], TAU)
s, y, did = score.numpy(), b["y_ret"].numpy(), b["date_id"].numpy()
man = []
for k in np.unique(did):
    g = s[did == k]
    q = np.quantile(g, 1.0 - FRAC)
    sdv = g.std() + 1e-8
    w = 1.0 / (1.0 + np.exp(-(g - q) / (TAU * sdv)))
    w = w / (w.sum() + 1e-8)
    if (w < 0).any() or abs(w.sum() - 1) > 1e-6:
        fails.append("T2b: weights not long-only simplex")
    man.append(float((w * y[did == k]).sum()))
man = np.array(man)
d2 = float(np.abs(port_t.numpy() - man).max())
if d2 > 1e-5:
    fails.append(f"T2: parity off max|d|={d2:.2e}")

# T5 net-of-cost turnover: net <= gross (cost only subtracts); identical
# weights across dates => ~0 turnover after the first rebalance
COSTt = 0.001
gross, _ = portfolio_returns(score, b["y_ret"], b["date_id"], TAU)
net, _ = portfolio_returns(score, b["y_ret"], b["date_id"], TAU,
                           sym_id=b["sym_id"], cost=COSTt)
if (net[1:] > gross[1:] + 1e-7).any():
    fails.append("T5a: net>gross after t0 (cost must subtract)")
# re-jury-6 fix: FIRST rebalance is burn-in (no initial-deployment cost)
if abs(float(net[0] - gross[0])) > 1e-7:
    fails.append("T5d: first-rebalance turnover charged (burn-in expected)")
# synthetic: same score & sym every date -> stable holdings, ~0 turnover
nd_s = 5
sc = torch.randn(50).repeat(nd_s)
dd = torch.arange(nd_s).repeat_interleave(50)
syq = torch.arange(50).repeat(nd_s)
gp, _ = portfolio_returns(sc, sc * 0 + 0.01, dd, TAU)
npp, _ = portfolio_returns(sc, sc * 0 + 0.01, dd, TAU,
                           sym_id=syq, cost=COSTt)
if float((gp[1:] - npp[1:]).abs().max()) > 1e-6:
    fails.append("T5c: stable holdings still charged turnover after t0")

# T6 HARD top-decile endpoint (PREREG §12 k.2 — what H1 is scored on)
from heads import hard_top_decile_returns, FRAC as _FR
hp, hu = hard_top_decile_returns(score, b["y_ret"], b["date_id"])
import numpy as _np
for k in _np.unique(did):
    g = score.numpy()[did == k]; ys = b["y_ret"].numpy()[did == k]
    kk = max(1, int(_np.ceil(_FR * len(g))))
    exp = ys[_np.argsort(-g)[:kk]].mean()
    got = float(hp[_np.where(hu.numpy() == k)[0][0]])
    if abs(got - exp) > 1e-5:
        fails.append(f"T6: hard-decile endpoint mismatch ({got:.5f}!={exp:.5f})")
        break
if len(hp) != len(_np.unique(did)):
    fails.append("T6b: hard-endpoint not one return per date")

# T3 ENFORCED per-symbol trading-row non-overlap (>= H rows) -- NO waiver
maxgapfail = 0
sel_set = set(pd.Series(sd).astype("datetime64[ns]"))
for sym in syms:
    idx = L.idx_of.get(sym, {})
    rows = sorted(idx[np.datetime64(d)] for d in sd
                  if np.datetime64(d) in idx)
    if len(rows) >= 2:
        g = np.diff(rows)
        if g.min() < H:
            maxgapfail += 1
if maxgapfail:
    fails.append(f"T3: {maxgapfail} symbols have consecutive rebalance "
                 f"dates < {H} trading rows apart (label overlap)")

# T4 operator-limit: PROVE w -> §4 hard top-decile equal-weight as tau->0
# (monotone convergence, not a single arbitrary cutoff).
torch.manual_seed(0)
N = 200
sc = torch.randn(N)
did0 = torch.zeros(N, dtype=torch.long)
k = int(np.ceil(FRAC * N))
topk = torch.topk(sc, k).indices
mass_seq = []
for tau in (1e-2, 1e-3, 1e-4, 1e-5):
    w = soft_top_decile(sc, did0, tau=tau)
    mass_seq.append(float(w[topk].sum()))
    if abs(float(w.sum()) - 1) > 1e-5:
        fails.append(f"T4d: not budget-normalized at tau={tau}")
w0 = soft_top_decile(sc, did0, tau=1e-5)
support = int((w0 > 1e-6).sum())
eqw_cv = float(w0[topk].std() / (w0[topk].mean() + 1e-12))
mono = all(mass_seq[i] <= mass_seq[i + 1] + 1e-9
           for i in range(len(mass_seq) - 1))
if not mono:
    fails.append(f"T4a: mass NOT monotone increasing as tau->0: {mass_seq}")
if mass_seq[-1] < 0.999:
    fails.append(f"T4b: tau=1e-5 top-decile mass={mass_seq[-1]:.5f} (<0.999)")
if not (k - 1 <= support <= k + 1):
    fails.append(f"T4c: tau=1e-5 support={support}, expected {k}")
if eqw_cv > 1e-3:
    fails.append(f"T4e: top-decile not equal-weight (cv={eqw_cv:.4f})")

print(f"[t] rebalance dates: {len(sd)} "
      f"({pd.Timestamp(sd.min()).date()}..{pd.Timestamp(sd.max()).date()})")
print(f"[t] first batch: {nd} dates, stocks/date "
      f"min={int(per.min())} max={int(per.max())}")
print(f"[t] T2 soft-decile parity max|d|={d2:.2e} (<1e-5)")
print(f"[t] T3 per-symbol min-row-gap fails={maxgapfail} (need 0)")
print(f"[t] T4 tau->0 mass seq (1e-2..1e-5)="
      f"{[round(m,4) for m in mass_seq]} support={support}(~{k}) "
      f"eqw_cv={eqw_cv:.1e} monotone={mono}")
print("RESULT:", "ALL PASS" if not fails else f"FAIL -> {fails}")
