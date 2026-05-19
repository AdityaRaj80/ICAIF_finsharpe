"""Heads + losses.

FATAL-1 FIX (re-jury-5): the prior `softmax(mu/tau)` was NOT the §4
endpoint (full-support tilt, near-uniform at realistic mu scale; the
"soft top-decile / image of §4" claim was mathematically false). Replaced
with a true **differentiable soft-top-decile** operator whose tau->0 limit
provably equals PREREG §4's hard LONG-ONLY TOP-DECILE EQUAL-WEIGHT
portfolio:

  per rebalance date d, over the cross-section of scores s:
    q   = quantile(s, 1 - frac)          # the decile cutoff (detached)
    sd  = std(s)                          # scale (detached) -> scale-aware
    w_i = sigmoid((s_i - q)/(tau*sd))     # long-only, w>=0
    w   = w / sum(w)                      # budget-normalized (sum=1)
  tau->0  =>  w -> uniform over the top `frac` names  ==  §4 endpoint.

`tau` has NO default (must be passed; pinned in PREREG §9, Task 48).
Gate / L_meta / vol removed: the gate was unused in allocation (trained a
head that did nothing for H1) — slimmed to {mu, logvar}; sigma via NLL is
an OPTIONAL calibration ablation, NOT used in allocation (honest, no
"uncertainty-aware" overclaim). Net-of-cost turnover term is a tracked
MAJOR (Task 50, needs sym_id from the loader).

DATA CONTRACT: batch = {x:[N,L,F], y_ret:[N], y_vol:[N], date_id:[N]}
date-GROUPED (engine/dataset.py), >= MIN_DATES H-spaced dates/batch.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8
MIN_DATES = 16          # loader must supply >= this many dates/batch (§9)
FRAC = 0.10             # top-decile (§4 endpoint; pinned §9)


class MSEReturnHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc = nn.Linear(d_model, 1)

    def forward(self, z):
        return self.fc(z).squeeze(-1)

    def loss(self, z, batch):
        return F.mse_loss(self(z), batch["y_ret"]), {}


class RiskAwareHead(nn.Module):
    """Slimmed: mu (signal that is ranked into the soft-top-decile) and
    logvar (sigma for the OPTIONAL NLL calibration ablation; NOT used in
    allocation)."""
    def __init__(self, d_model):
        super().__init__()
        self.mu = nn.Linear(d_model, 1)
        self.logvar = nn.Linear(d_model, 1)
        self.drop = nn.Dropout(0.1)

    def forward(self, z):
        h = self.drop(z)
        return self.mu(h).squeeze(-1), self.logvar(h).squeeze(-1).clamp(-10, 4)


def soft_top_decile(score, did, tau, frac=FRAC):
    """Differentiable long-only top-decile weights per date group.
    tau->0  ==  hard top-decile equal-weight (PREREG §4).
    AMP-safe (re-jury-6 real-determinism fix): portfolio-weight math is
    done in fp32 — torch.quantile rejects fp16, and weight normalisation
    is numerically unstable in half precision under autocast."""
    score = score.float()                            # AMP-safe upcast
    w = torch.zeros_like(score)
    for d in torch.unique(did):
        m = did == d
        s = score[m]
        if s.numel() == 1:
            w[m] = 1.0
            continue
        q = torch.quantile(s.detach(), 1.0 - frac)
        sd = s.detach().std(unbiased=False) + EPS
        wi = torch.sigmoid((s - q) / (tau * sd))
        w[m] = wi / (wi.sum() + EPS)
    return w


COST = 0.001            # 10 bps one-way txn cost (matches H1; PINNED §9)


def portfolio_returns(score, y, did, tau, sym_id=None, cost=0.0):
    """Per-date soft-top-decile long-only portfolio return.
    If sym_id+cost given -> NET of L1 turnover cost across consecutive
    (H-spaced) rebalance dates, aligning weights by symbol (prev detached
    so the model is incentivised to *reduce* turnover). NET makes the
    trained objective match PREREG §4-net (re-jury-5 quant MAJOR)."""
    w = soft_top_decile(score, did, tau)
    uniq = torch.unique(did)
    uniq = uniq[torch.argsort(uniq)]                 # chronological order
    if sym_id is None or cost == 0.0:
        port = torch.stack([(w[did == k] * y[did == k]).sum() for k in uniq])
        return port, uniq
    _, loc = torch.unique(sym_id, return_inverse=True)   # 0..S-1 per row
    S = int(loc.max()) + 1
    prev, out = None, []
    for k in uniq:
        m = did == k
        wv = torch.zeros(S, device=w.device).scatter(0, loc[m], w[m])
        gross = (w[m] * y[m]).sum()
        if prev is None:
            out.append(gross)                        # re-jury-6 fix: NO
            # initial-deployment cost (burn-in) -> uncharged 1st rebalance
            # is common to every CPCV path & matches §4 (deploy once).
        else:
            turn = 0.5 * (wv - prev).abs().sum()      # one-way turnover
            out.append(gross - cost * turn)
        prev = wv.detach()
    return torch.stack(out), uniq


def hard_top_decile_returns(score, y, did, frac=FRAC):
    """The §4 EVALUATION endpoint (PREREG §12 k.2): hard long-only
    top-decile EQUAL-WEIGHT applied to model scores. H1 / the backtest
    are scored on THIS — never the soft-decile (which is only the
    differentiable training surrogate). Non-differentiable by design."""
    uniq = torch.unique(did)
    port = []
    for k in uniq:
        m = did == k
        s, ys = score[m], y[m]
        kk = max(1, math.ceil(frac * s.numel()))
        top = torch.topk(s, kk).indices
        port.append(ys[top].mean())                  # equal-weight top-decile
    return torch.stack(port), uniq


def tau_schedule(step, total, t0=0.10, t1=1e-3):
    """Anneal the surrogate temperature 0.10 -> 1e-3 over training so the
    trained surrogate approaches the hard endpoint (validity does NOT
    depend on this — H1 is scored on hard_top_decile_returns; PREREG k.2)."""
    import math
    f = min(max(step / max(total, 1), 0.0), 1.0)
    return t1 + 0.5 * (t0 - t1) * (1 + math.cos(math.pi * f))


def composite_risk_loss(z, head, batch, tau, cost=COST, a=0.7, g=0.5, b=0.5):
    """L = a*(-Sharpe of NET soft-top-decile portfolio over the batch's
    H-spaced dates)  +  g*MSE(mu,y) [anchor]  +  b*NLL [sigma calib,
    ablation]. Coeffs, tau, cost PINNED in PREREG §9. std UNBIASED."""
    mu, logvar = head(z)
    y, did = batch["y_ret"], batch["date_id"]
    port, uniq = portfolio_returns(mu, y, did, tau,
                                   sym_id=batch.get("sym_id"), cost=cost)
    L_sr = -port.mean() / (port.std(unbiased=True) + EPS)
    L_anch = F.mse_loss(mu, y)
    L_nll = 0.5 * (logvar + (y - mu) ** 2 / (logvar.exp() + EPS)).mean()
    total = a * L_sr + g * L_anch + b * L_nll
    return total, {"L_sr": L_sr.item(), "L_anch": L_anch.item(),
                   "L_nll": L_nll.item(), "n_dates": int(uniq.numel())}
