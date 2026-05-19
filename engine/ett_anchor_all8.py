"""All-8 native-head ETTh1 anchor (PREREG §12 k.5).

Trio {DLinear,PatchTST,iTransformer} use their special native heads
(from ett_anchor_native); the other 5 use the standard LTSF
projection-from-encoder native head (the canonical way these baselines
forecast). PASS gate per arch = 1.5x cited-worst-published ETTh1-96 MSE
(mechanically derived, fixed a-priori in k.5) + beats persistence +
non-pathological. Writes bench/ett_anchor_all8_report.md.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from determinism import seed_all
from ett_anchor_native import (make_split, RevIN, run, DLinearNative,
                               PatchTSTNative, iTransformerNative, SEQ,
                               PRED, DM, C)
from models import REGISTRY

# k.5 frozen gates = 1.5 x cited worst published ETTh1-96 MSE
GATE = {"dlinear": .60, "patchtst": .62, "itransformer": .59,
        "gcformer": .68, "tft": .90, "cnn": .83, "lstm": 1.05, "rnn": 1.05}


class ProjNative(nn.Module):
    """Standard LTSF native head: encoder rep -> linear to PRED*C + RevIN
    (the canonical forecaster for TFT/GCformer/LSTM/RNN/TCN)."""
    def __init__(self, name):
        super().__init__()
        self.enc = REGISTRY[name](C, SEQ, DM)
        self.proj = nn.Linear(DM, PRED * C)
        self.rev = RevIN()

    def forward(self, x):
        xn = self.rev.fwd(x)
        return self.rev.inv(self.proj(self.enc(xn)).view(-1, PRED, C))


def builder(name):
    if name == "dlinear":
        return DLinearNative
    if name == "patchtst":
        return PatchTSTNative
    if name == "itransformer":
        return iTransformerNative
    return lambda: ProjNative(name)


if __name__ == "__main__":
    t0 = time.time()
    tr, va, te = make_split()
    Xte, Yte = te
    pers = nn.functional.mse_loss(Xte[:, -1:, :].repeat(1, PRED, 1),
                                  Yte).item()
    print(f"[a8] persistence_MSE={pers:.4f}", flush=True)
    res, fails = {}, []
    for nm in ["dlinear", "patchtst", "itransformer", "gcformer", "tft",
               "cnn", "lstm", "rnn"]:
        mse, mae, ep = run(builder(nm), tr, va, te)
        res[nm] = mse
        print(f"[a8] {nm:<13} MSE={mse:.4f} MAE={mae:.4f} ep={ep} gate<"
              f"{GATE[nm]} {time.time()-t0:,.0f}s", flush=True)
    tmin = min(res.values())
    R = ["# ETTh1 All-8 Native Anchor (PREREG §12 k.5)\n",
         f"_Native heads + RevIN, d_model={DM}, seed 0; gates = "
         f"1.5x cited-worst-published (mechanically derived, fixed "
         f"a-priori). persistence={pers:.4f}._\n",
         "| model | MSE | gate | <gate | <pers | <=2x best |",
         "|--|--|--|--|--|--|"]
    for nm, v in res.items():
        g_ok, p_ok, b_ok = v < GATE[nm], v < pers, v <= 2 * tmin
        if not (np.isfinite(v) and g_ok and p_ok and b_ok):
            fails.append(nm)
        R.append(f"| {nm} | {v:.4f} | {GATE[nm]} | {'Y' if g_ok else 'N'} "
                 f"| {'Y' if p_ok else 'N'} | {'Y' if b_ok else 'N'} |")
    verdict = "PASS" if not fails else f"FAIL ({fails})"
    R.append(f"\n**ALL-8 ANCHOR VERDICT: {verdict}** (gates pre-registered "
             f"PREREG §12 k.5, hash amendment k, BEFORE this run). "
             f"elapsed {time.time()-t0:,.0f}s")
    open(r"D:\Study\FinSharpe\bench\ett_anchor_all8_report.md", "w",
         encoding="utf-8").write("\n".join(R))
    print("ALL8-ANCHOR:", verdict, flush=True)
