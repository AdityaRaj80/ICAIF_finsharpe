"""Native-head ETTh1 anchor (PREREG §9a-NATIVE, amendment j).

Literature-valid fidelity test: each fidelity-critical model uses its
NATIVE forecasting head (not the FinSharpe encoder-pool) + RevIN — the
config in which published ETTh1 numbers were obtained. Detects real impl
bugs/impostors. PASS (pre-registered, amendment j): trio
{DLinear,PatchTST,iTransformer} finite, < persistence, test MSE < 0.55,
non-pathological ordering. Writes bench/ett_anchor_native_report.md.
"""
import os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from determinism import seed_all
from models import PatchTST as _PatchTST, iTransformer as _iT, _MovingAvg

CSV = r"D:\Study\FinSharpe\bench\ETTh1.csv"
OUT = r"D:\Study\FinSharpe\bench\ett_anchor_native_report.md"
SEQ, PRED, DM, BS, EPOCHS, PAT = 96, 96, 128, 64, 15, 3
DEV = "cuda" if torch.cuda.is_available() else "cpu"
C = 7


def make_split():
    df = pd.read_csv(CSV).drop(columns=["date"]).astype("float32").values
    b1 = [0, 12 * 30 * 24 - SEQ, 16 * 30 * 24 - SEQ]
    b2 = [12 * 30 * 24, 16 * 30 * 24, 20 * 30 * 24]
    mu, sd = df[:b2[0]].mean(0, keepdims=True), df[:b2[0]].std(0, keepdims=True) + 1e-8
    d = (df - mu) / sd

    def win(a, b):
        xs, ys = [], []
        for i in range(a, b - SEQ - PRED + 1):
            xs.append(d[i:i + SEQ]); ys.append(d[i + SEQ:i + SEQ + PRED])
        return torch.tensor(np.stack(xs)), torch.tensor(np.stack(ys))
    return win(*[b1[0], b2[0]]), win(b1[1], b2[1]), win(b1[2], b2[2])


class RevIN(nn.Module):
    """Per-sample per-series instance norm; de-norm the forecast."""
    def fwd(self, x):
        self.m = x.mean(1, keepdim=True)
        self.s = x.std(1, keepdim=True) + 1e-5
        return (x - self.m) / self.s

    def inv(self, y):
        return y * self.s + self.m


class DLinearNative(nn.Module):              # canonical (Zeng AAAI'23)
    def __init__(self):
        super().__init__()
        self.dec = _MovingAvg(25)
        self.lt = nn.Linear(SEQ, PRED)
        self.ls = nn.Linear(SEQ, PRED)
        self.rev = RevIN()

    def forward(self, x):                    # [B,L,C]
        x = self.rev.fwd(x).transpose(1, 2)  # [B,C,L]
        t = self.dec(x); s = x - t
        y = (self.lt(t) + self.ls(s)).transpose(1, 2)   # [B,PRED,C]
        return self.rev.inv(y)


class PatchTSTNative(nn.Module):             # channel-indep -> per-chan head
    def __init__(self):
        super().__init__()
        self.bb = _PatchTST(C, SEQ, DM)
        self.proj = nn.Linear(DM, PRED)      # shared across channels
        self.rev = RevIN()

    def forward(self, x):
        xn = self.rev.fwd(x)
        r = self.bb.backbone(xn)             # [B,C,d] channel-independent
        return self.rev.inv(self.proj(r).transpose(1, 2))   # [B,PRED,C]


class iTransformerNative(nn.Module):         # variate tokens -> projection
    def __init__(self):
        super().__init__()
        self.it = _iT(C, SEQ, DM)
        self.proj = nn.Linear(DM, PRED)
        self.rev = RevIN()

    def forward(self, x):
        xn = self.rev.fwd(x)
        tok = self.it.tr(self.it.tokens(xn))  # [B,C,d] (post variate-attn)
        return self.rev.inv(self.proj(tok).transpose(1, 2))


def run(M, tr, va, te):
    seed_all(0)
    m = M().to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    Xtr, Ytr = tr; Xva, Yva = va; Xte, Yte = te
    best, bad, bs = 1e9, 0, None
    for ep in range(EPOCHS):
        m.train(); idx = torch.randperm(len(Xtr))
        for s in range(0, len(idx), BS):
            b = idx[s:s + BS]; opt.zero_grad()
            nn.functional.mse_loss(m(Xtr[b].to(DEV)),
                                   Ytr[b].to(DEV)).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vp = torch.cat([m(Xva[i:i + 256].to(DEV)).cpu()
                            for i in range(0, len(Xva), 256)])
        vm = nn.functional.mse_loss(vp, Yva).item()
        if vm < best - 1e-5:
            best, bad = vm, 0
            bs = {k: v.cpu().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= PAT:
                break
    m.load_state_dict(bs); m.eval()
    with torch.no_grad():
        tp = torch.cat([m(Xte[i:i + 256].to(DEV)).cpu()
                        for i in range(0, len(Xte), 256)])
    return (nn.functional.mse_loss(tp, Yte).item(),
            (tp - Yte).abs().mean().item(), ep + 1)


if __name__ == "__main__":
    t0 = time.time()
    tr, va, te = make_split()
    Xte, Yte = te
    pers = nn.functional.mse_loss(Xte[:, -1:, :].repeat(1, PRED, 1),
                                  Yte).item()
    print(f"[ettN] dev={DEV} persistence_MSE={pers:.4f}", flush=True)
    M = {"dlinear": DLinearNative, "patchtst": PatchTSTNative,
         "itransformer": iTransformerNative}
    res = {}
    for nm, cls in M.items():
        mse, mae, ep = run(cls, tr, va, te)
        res[nm] = mse
        print(f"[ettN] {nm:<13} test_MSE={mse:.4f} MAE={mae:.4f} ep={ep} "
              f"{time.time()-t0:,.0f}s", flush=True)
    tmin = min(res.values())
    fails = [n for n, v in res.items()
             if not (np.isfinite(v) and v < pers and v < 0.55
                     and v <= 2 * tmin)]
    verdict = "PASS" if not fails else f"FAIL ({fails})"
    R = ["# ETTh1 Native-Head Anchor (PREREG §9a-NATIVE, amendment j)\n",
         f"_Informer split 96->96, native heads + RevIN, d_model={DM}, "
         f"seed 0. persistence MSE={pers:.4f}. Published ETTh1 96 ~"
         f"0.37-0.42 (full tuning); PASS gate <0.55._\n",
         "| model (native) | test MSE | <pers | <0.55 | <=2x trio-best |",
         "|--|--|--|--|--|"]
    for n, v in res.items():
        R.append(f"| {n} | {v:.4f} | {'Y' if v<pers else 'N'} | "
                 f"{'Y' if v<0.55 else 'N'} | {'Y' if v<=2*tmin else 'N'} |")
    R.append(f"\n**ANCHOR-NATIVE VERDICT: {verdict}** "
             f"(pre-registered amendment j, hash before run). "
             f"elapsed {time.time()-t0:,.0f}s")
    open(OUT, "w", encoding="utf-8").write("\n".join(R))
    print("ANCHOR-NATIVE:", verdict, "->", OUT, flush=True)
