"""H1 / backtest scorer — the §4 EVALUATION path (closes re-jury-7 FATAL).

This is the executable scoring pipeline (not prose): given a trained
model's per-(date,stock) test scores, it forms the portfolio with
`hard_top_decile_returns` (the §4 HARD long-only top-decile endpoint,
PREREG §12 k.2 — NEVER the soft surrogate), net of burn-in turnover,
then computes the paired contrast vs the MSE-arm and Ridge:
  - annualised Sharpe per arm + paired difference series
  - Deflated Sharpe Ratio of the difference (Bailey & Lopez de Prado 2014),
    deflated by N=1152 trials (PREREG §12 k.3)
  - PBO via CSCV (Lopez de Prado 2016), 252 paths (PREREG §12 k.4)
  - rank-IC per arm + Delta rank-IC
  - the pre-registered k.6 accept rule.
score_h1() is THE H1 scorer; the self-test asserts the PIPELINE routes
through the hard endpoint and never `soft_top_decile`.
"""
import math
import numpy as np
import torch
from scipy.stats import norm, spearmanr

from heads import hard_top_decile_returns, FRAC, COST
from dataset import cscv_folds

ANN = 252.0 / 5.0          # H=5 -> ~50 non-overlapping periods / yr
DSR_N = 1152               # PREREG §12 k.3
EUL = 0.5772156649


def _net_hard_port(score, y, did, sym_id, cost=COST):
    """Per-period return of the HARD top-decile endpoint, net of one-way
    turnover with an UNCHARGED first rebalance (burn-in; PREREG k.2 / the
    turnover fix). Long-only equal-weight top-decile by model score."""
    score = np.asarray(score); y = np.asarray(y)
    did = np.asarray(did); sym_id = np.asarray(sym_id)
    uniq = np.unique(did)
    prevw, out = None, []
    syms = np.unique(sym_id)
    sidx = {s: i for i, s in enumerate(syms)}
    for d in uniq:
        m = did == d
        s, ys, ss = score[m], y[m], sym_id[m]
        kk = max(1, math.ceil(FRAC * len(s)))
        top = np.argsort(-s)[:kk]
        w = np.zeros(len(syms))
        w[[sidx[x] for x in ss[top]]] = 1.0 / kk          # equal-weight
        gross = ys[top].mean()
        if prevw is None:
            out.append(gross)                              # burn-in
        else:
            out.append(gross - cost * 0.5 * np.abs(w - prevw).sum())
        prevw = w
    return np.array(out)


def _sharpe(r):
    r = np.asarray(r, float)
    return r.mean() / (r.std(ddof=1) + 1e-12) * math.sqrt(ANN)


def deflated_sharpe(diff, n_trials=DSR_N):
    """DSR of the difference series (Bailey & Lopez de Prado 2014).
    Returns (annualised_diff_Sharpe, DSR_prob). p = 1 - DSR_prob."""
    r = np.asarray(diff, float)
    n = len(r)
    sr = r.mean() / (r.std(ddof=1) + 1e-12)                # per-period
    g3 = float(((r - r.mean()) ** 3).mean() / (r.std() ** 3 + 1e-12))
    g4 = float(((r - r.mean()) ** 4).mean() / (r.std() ** 4 + 1e-12))
    # expected max Sharpe under N independent trials (variance ~ 1/n)
    sr_var = 1.0 / max(n - 1, 1)
    sr0 = math.sqrt(sr_var) * ((1 - EUL) * norm.ppf(1 - 1.0 / n_trials)
                               + EUL * norm.ppf(1 - 1.0 / (n_trials * math.e)))
    denom = math.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr * sr, 1e-9))
    dsr = float(norm.cdf((sr - sr0) * math.sqrt(max(n - 1, 1)) / denom))
    return sr * math.sqrt(ANN), dsr


def cscv_pbo(perf_matrix, dates):
    """PBO via CSCV (Lopez de Prado 2016): perf_matrix [n_periods x
    n_configs]. Fraction of S=10 C(10,5)=252 splits where the IS-best
    config is below the OOS median (logit<=0)."""
    M = np.asarray(perf_matrix, float)
    folds = cscv_folds(np.asarray(dates), S=10, purge=0)   # 252 recombos
    pos = {d: i for i, d in enumerate(np.sort(np.asarray(dates)))}
    lam = []
    for tr, te in folds:
        ti = [pos[d] for d in tr]; oi = [pos[d] for d in te]
        if not ti or not oi:
            continue
        is_perf = M[ti].mean(0); oos = M[oi].mean(0)
        nstar = int(np.argmax(is_perf))
        rank = (oos.argsort().argsort()[nstar] + 1) / (len(oos) + 1)
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        lam.append(math.log(rank / (1 - rank)))
    lam = np.array(lam)
    return float((lam <= 0).mean()) if len(lam) else float("nan")


def score_h1(risk_scores, mse_scores, ridge_scores, y, did, sym_id):
    """THE H1 scorer. All three arms scored on the SAME hard top-decile
    endpoint. Returns the paired contrast + the k.6 accept booleans."""
    pr = _net_hard_port(risk_scores, y, did, sym_id)
    pm = _net_hard_port(mse_scores, y, did, sym_id)
    pg = _net_hard_port(ridge_scores, y, did, sym_id)
    d_vs_mse, d_vs_rdg = pr - pm, pr - pg
    out = {"sharpe_risk": _sharpe(pr), "sharpe_mse": _sharpe(pm),
           "sharpe_ridge": _sharpe(pg)}
    for tag, dd in (("vs_mse", d_vs_mse), ("vs_ridge", d_vs_rdg)):
        ds, dp = deflated_sharpe(dd)
        out[f"dSharpe_{tag}"] = ds
        out[f"DSR_{tag}"] = dp
        out[f"p_{tag}"] = 1 - dp
    ic_r = spearmanr(risk_scores, y)[0]
    ic_m = spearmanr(mse_scores, y)[0]
    out["dRankIC_vs_mse"] = float(ic_r - ic_m)
    uniq = np.unique(did)
    M = np.column_stack([
        [(_net_hard_port(s[did == d], y[did == d], did[did == d],
                         sym_id[did == d])[0]) for d in uniq]
        for s in (risk_scores, mse_scores, ridge_scores)])
    out["PBO"] = cscv_pbo(M, uniq)
    # PREREG §12 k.6 accept rule (operationalised: p<0.05 == DSR>0.95)
    out["H1_ACCEPT"] = bool(
        out["dSharpe_vs_mse"] >= 0.20 and out["dSharpe_vs_ridge"] >= 0.20
        and out["dRankIC_vs_mse"] >= 0.01 and out["PBO"] <= 0.5
        and out["p_vs_mse"] < 0.05 and out["p_vs_ridge"] < 0.05)
    out["_endpoint"] = "hard_top_decile_returns"        # provenance tag
    return out


if __name__ == "__main__":
    # SELF-TEST: prove the PIPELINE routes through the HARD endpoint
    # (not soft_top_decile) and the stats behave. No trained model needed.
    import inspect
    import heads
    src = inspect.getsource(score_h1) + inspect.getsource(_net_hard_port)
    fails = []
    if "hard_top_decile_returns" not in inspect.getsource(_net_hard_port) \
       and "argsort(-s)" not in inspect.getsource(_net_hard_port):
        fails.append("scorer not using hard top-decile selection")
    if "soft_top_decile" in src:
        fails.append("FATAL: scorer references the soft surrogate")
    # parity: _net_hard_port top-decile == heads.hard_top_decile_returns
    rng = np.random.default_rng(0)
    nD, nS = 60, 80
    did = np.repeat(np.arange(nD), nS)
    sym = np.tile(np.arange(nS), nD)
    sc = rng.standard_normal(nD * nS)
    y = 0.02 * rng.standard_normal(nD * nS) + 0.001 * sc      # mild signal
    hp, _ = hard_top_decile_returns(torch.tensor(sc), torch.tensor(y),
                                    torch.tensor(did))
    mine = _net_hard_port(sc, y, did, sym, cost=0.0)          # gross
    parity = float(np.abs(hp.numpy() - mine).max())
    if parity > 1e-6:
        fails.append(f"hard-endpoint parity off: {parity:.2e}")
    # signal arm should beat noise arm; identical arms -> not accepted
    noise = rng.standard_normal(nD * nS)
    r = score_h1(sc, noise, noise, y, did, sym)
    same = score_h1(sc, sc, sc, y, did, sym)
    if not (r["sharpe_risk"] > r["sharpe_mse"]):
        fails.append("signal arm did not beat noise arm")
    if same["H1_ACCEPT"]:
        fails.append("identical arms wrongly ACCEPTED H1")
    if not (0.0 <= r["PBO"] <= 1.0):
        fails.append(f"PBO out of range: {r['PBO']}")
    print(f"[bt] hard-endpoint parity max|d|={parity:.2e}")
    print(f"[bt] signal vs noise: SR_risk={r['sharpe_risk']:.2f} "
          f"SR_mse={r['sharpe_mse']:.2f} dSharpe_vs_mse="
          f"{r['dSharpe_vs_mse']:.3f} PBO={r['PBO']:.3f} "
          f"H1_ACCEPT={r['H1_ACCEPT']}")
    print(f"[bt] identical-arms H1_ACCEPT={same['H1_ACCEPT']} (must be False)")
    print("BACKTEST-SELFTEST:", "ALL PASS" if not fails else f"FAIL {fails}")
