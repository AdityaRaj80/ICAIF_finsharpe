"""Real-training-path determinism proof (re-jury-6 MAJOR).

Not a toy: runs the ACTUAL Phase-2 path twice — DateGroupedLoader batches
+ AMP autocast + GradScaler + Adam + composite_risk_loss + 2 epochs, on a
transformer (iTransformer) and a conv (cnn) backbone — and asserts the
final weights are BIT-IDENTICAL across the two seeded runs. Supports the
PREREG §7 "bit-reproducible per seed" claim on the real path.
Writes bench/determinism_real_report.md.
"""
import os, sys, time
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from determinism import seed_all
from dataset import DateGroupedLoader
from registry import Model, TAU

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SYMS = pd.read_csv(r"D:\Study\FinSharpe\universe\tier1.txt",
                   header=None)[0].astype(str).str.strip().tolist()[:30]


def one_run(name):
    seed_all(0)
    L = DateGroupedLoader("test", 5, symbols=SYMS, min_dates=8)
    m = Model(name, "risk", n_feat=len(L.feat), seq_len=504,
              d_model=64).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler(DEV, enabled=(DEV == "cuda"))
    for _ep in range(2):
        for b in L.batches():
            b = {k: v.to(DEV) for k, v in b.items()}
            opt.zero_grad()
            with torch.amp.autocast(DEV, enabled=(DEV == "cuda")):
                loss, _ = m.loss(b)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
    return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}


if __name__ == "__main__":
    t0 = time.time()
    rep = ["# Real-Path Determinism (re-jury-6 MAJOR)\n",
           f"_2 epochs, real loader+AMP+GradScaler+Adam+composite loss, "
           f"dev={DEV}, seed 0 x2 -> bit-identical final weights._\n",
           "| backbone | params compared | max_absdiff | bit-identical |",
           "|--|--|--|--|"]
    allok = True
    for nm in ["itransformer", "cnn"]:
        a = one_run(nm)
        b = one_run(nm)
        md = max(float((a[k] - b[k]).abs().max())
                 for k in a if a[k].is_floating_point())
        ident = md == 0.0
        allok &= ident
        rep.append(f"| {nm} | {len(a)} | {md:.2e} | "
                   f"{'YES' if ident else 'NO'} |")
        print(f"[det] {nm} max_absdiff={md:.2e} bit-identical={ident} "
              f"{time.time()-t0:,.0f}s", flush=True)
    rep.append(f"\n**REAL-DETERMINISM: {'PASS' if allok else 'FAIL'}** "
               f"(real Phase-2 path; elapsed {time.time()-t0:,.0f}s)")
    open(r"D:\Study\FinSharpe\bench\determinism_real_report.md", "w",
         encoding="utf-8").write("\n".join(rep))
    print("REAL-DETERMINISM:", "PASS" if allok else "FAIL", flush=True)
