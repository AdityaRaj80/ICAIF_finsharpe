"""External-validity anchor: standard ETTh1 forecasting (PREREG §9a).

Impostor/bug detector for the in-house backbones (NOT SOTA reproduction).
Informer split (12/4/4 mo), input 96 -> predict 96, all 7 vars,
train-fit per-feature standardisation, encoder + linear forecast head.
PASS criteria fixed in PREREG §9a (pre-registered before this ran).
Writes bench/ett_anchor_report.md.
"""
import os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from determinism import seed_all
from models import REGISTRY

CSV = r"D:\Study\FinSharpe\bench\ETTh1.csv"
OUT = r"D:\Study\FinSharpe\bench\ett_anchor_report.md"
SEQ, PRED, DM, BS, EPOCHS, PAT = 96, 96, 128, 64, 12, 3
MODELS = ["dlinear", "patchtst", "itransformer", "gcformer", "tft",
          "cnn", "lstm", "rnn"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def make_split():
    df = pd.read_csv(CSV).drop(columns=["date"]).astype("float32").values
    b1 = [0, 12 * 30 * 24 - SEQ, 16 * 30 * 24 - SEQ]      # Informer borders
    b2 = [12 * 30 * 24, 16 * 30 * 24, 20 * 30 * 24]
    mu = df[b1[0]:b2[0]].mean(0, keepdims=True)
    sd = df[b1[0]:b2[0]].std(0, keepdims=True) + 1e-8
    d = (df - mu) / sd
    def win(a, b):
        xs, ys = [], []
        for i in range(a, b - SEQ - PRED + 1):
            xs.append(d[i:i + SEQ]); ys.append(d[i + SEQ:i + SEQ + PRED])
        return (torch.tensor(np.stack(xs)), torch.tensor(np.stack(ys)))
    return win(b1[0], b2[0]), win(b1[1], b2[1]), win(b1[2], b2[2])


class Fc(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.enc = REGISTRY[name](7, SEQ, DM)
        self.head = nn.Linear(DM, PRED * 7)

    def forward(self, x):
        return self.head(self.enc(x)).view(-1, PRED, 7)


def run(name, tr, va, te):
    seed_all(0)
    m = Fc(name).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    Xtr, Ytr = tr; Xva, Yva = va; Xte, Yte = te
    best, bad, bstate = 1e9, 0, None
    for ep in range(EPOCHS):
        m.train()
        idx = torch.randperm(len(Xtr))
        for s in range(0, len(idx), BS):
            b = idx[s:s + BS]
            opt.zero_grad()
            loss = nn.functional.mse_loss(m(Xtr[b].to(DEV)), Ytr[b].to(DEV))
            loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vp = torch.cat([m(Xva[i:i + 256].to(DEV)).cpu()
                            for i in range(0, len(Xva), 256)])
            vm = nn.functional.mse_loss(vp, Yva).item()
        if vm < best - 1e-5:
            best, bad, bstate = vm, 0, {k: v.cpu().clone()
                                       for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= PAT:
                break
    m.load_state_dict(bstate)
    m.eval()
    with torch.no_grad():
        tp = torch.cat([m(Xte[i:i + 256].to(DEV)).cpu()
                        for i in range(0, len(Xte), 256)])
    mse = nn.functional.mse_loss(tp, Yte).item()
    mae = (tp - Yte).abs().mean().item()
    return mse, mae, ep + 1


if __name__ == "__main__":
    t0 = time.time()
    tr, va, te = make_split()
    Xte, Yte = te
    # persistence baseline: repeat last input step for PRED steps
    pers = nn.functional.mse_loss(
        Xte[:, -1:, :].repeat(1, PRED, 1), Yte).item()
    print(f"[ett] dev={DEV} train={len(tr[0])} val={len(va[0])} "
          f"test={len(Xte)} persistence_MSE={pers:.4f}", flush=True)
    rows, fails = [], []
    for nm in MODELS:
        try:
            mse, mae, eps = run(nm, tr, va, te)
            rows.append((nm, mse, mae, eps))
            print(f"[ett] {nm:<13} test_MSE={mse:.4f} MAE={mae:.4f} "
                  f"ep={eps} {time.time()-t0:,.0f}s", flush=True)
        except Exception as e:
            rows.append((nm, float("nan"), float("nan"), 0))
            print(f"[ett] {nm:<13} FAIL {type(e).__name__}: {e}", flush=True)
    dl = dict((r[0], r[1]) for r in rows).get("dlinear", float("nan"))
    R = ["# ETTh1 External-Validity Anchor (PREREG §9a)\n",
         f"_Informer split, 96->96, 7 vars, d_model={DM}, seed 0; "
         f"impostor/bug detector, NOT SOTA. persistence MSE={pers:.4f}._\n",
         "| model | test MSE | test MAE | epochs | <pers | <0.80 | "
         "<=2.5x DLinear |", "|--|--|--|--|--|--|--|"]
    for nm, mse, mae, eps in rows:
        ok_p = mse < pers
        ok_c = mse < 0.80
        ok_d = (nm == "dlinear") or (mse <= 2.5 * dl) or nm in ("lstm", "rnn")
        if not (np.isfinite(mse) and ok_p and ok_c and ok_d):
            fails.append(nm)
        R.append(f"| {nm} | {mse:.4f} | {mae:.4f} | {eps} | "
                 f"{'Y' if ok_p else 'N'} | {'Y' if ok_c else 'N'} | "
                 f"{'Y' if ok_d else 'N'} |")
    verdict = "PASS" if not fails else f"FAIL ({fails})"
    R.append(f"\n**ANCHOR VERDICT: {verdict}** "
             f"(criteria pre-registered PREREG §9a, hash amendment i). "
             f"LSTM/RNN exempt from the 2.5x rule (intrinsically weak "
             f"baselines, disclosed). elapsed {time.time()-t0:,.0f}s")
    open(OUT, "w", encoding="utf-8").write("\n".join(R))
    print("ANCHOR:", verdict, "-> wrote", OUT, flush=True)
