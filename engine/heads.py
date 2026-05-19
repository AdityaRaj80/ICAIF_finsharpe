"""Heads + losses.

Risk objective REBUILT (re-jury models F1/M1) to be the differentiable
IMAGE of the PREREG §4 endpoint:
  * per rebalance date, within-cross-section LONG-ONLY simplex weights
    w = softmax(score / tau)  (w>=0, sum=1 -> budget-normalized long-only;
    low tau == concentrate in top names == soft top-decile);
  * per-date portfolio return  p_d = sum_i w_i * r_i^H ;
  * Sharpe over the batch's H-SPACED dates (loader guarantees >= MIN_DATES
    non-overlapping dates/batch so this is a low-variance estimator that
    matches the §4 non-overlapping H-spaced backtest).
score = predicted mu (NO tanh(10*mu/sigma): that saturated to a sign
classifier — M1). sigma feeds NLL/vol only; gate is a detached
meta-label (Lopez de Prado) on sign(mu*y), not circular.

DATA CONTRACT: batch = {x:[N,L,F], y_ret:[N], y_vol:[N], date_id:[N]}
date-GROUPED (engine/dataset.py); >= MIN_DATES distinct H-spaced dates.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8
MIN_DATES = 16          # loader must supply >= this many dates/batch


class MSEReturnHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc = nn.Linear(d_model, 1)

    def forward(self, z):
        return self.fc(z).squeeze(-1)

    def loss(self, z, batch):
        return F.mse_loss(self(z), batch["y_ret"]), {}


class RiskAwareHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mu = nn.Linear(d_model, 1)
        self.logvar = nn.Linear(d_model, 1)
        self.vol = nn.Linear(d_model, 1)
        self.gate = nn.Linear(d_model, 1)
        self.drop = nn.Dropout(0.1)

    def forward(self, z):
        h = self.drop(z)
        return (self.mu(h).squeeze(-1),
                self.logvar(h).squeeze(-1).clamp(-10, 4),
                self.vol(h).squeeze(-1),
                torch.sigmoid(self.gate(h).squeeze(-1)))


def _segment_softmax(score, did, tau):
    """Long-only simplex weights within each date group (vectorised)."""
    s = score / tau
    m = torch.zeros_like(s)
    for d in torch.unique(did):
        mask = did == d
        m[mask] = torch.softmax(s[mask], dim=0)
    return m                                   # w>=0, sum_per_date=1


def portfolio_returns(score, y, did, tau):
    """Per-date long-only-simplex portfolio return (the §4-endpoint image).
    Exposed for independent unit-testing (engine/test_pipeline.py)."""
    w = _segment_softmax(score, did, tau)
    uniq = torch.unique(did)
    port = torch.stack([(w[did == k] * y[did == k]).sum() for k in uniq])
    return port, uniq


def composite_risk_loss(z, head, batch, tau=0.10,
                        a=0.7, b=0.5, g=0.5, d=0.3, e=0.1):
    """5-term composite; L_sr is the §4-endpoint image (see module doc).
    Coefficients (a,b,g,d,e) and tau are PINNED in PREREG §9."""
    mu, logvar, vol, gate = head(z)
    y, yv, did = batch["y_ret"], batch["y_vol"], batch["date_id"]
    port, uniq = portfolio_returns(mu, y, did, tau)
    # Sharpe over the batch's H-spaced dates (negative -> minimised)
    L_sr = -port.mean() / (port.std(unbiased=False) + EPS)

    sig2 = logvar.exp()
    L_nll = 0.5 * (logvar + (y - mu) ** 2 / (sig2 + EPS)).mean()
    L_anch = F.mse_loss(mu, y)                  # return anchor
    L_vol = F.mse_loss(vol, yv)
    meta = ((mu.detach() * y) > 0).float()      # detached meta-label on mu
    L_meta = F.binary_cross_entropy(gate.clamp(1e-4, 1 - 1e-4), meta)

    total = a * L_sr + b * L_nll + g * L_anch + d * L_vol + e * L_meta
    return total, {"L_sr": L_sr.item(), "L_nll": L_nll.item(),
                   "L_anch": L_anch.item(), "L_vol": L_vol.item(),
                   "L_meta": L_meta.item(), "n_dates": int(uniq.numel())}
