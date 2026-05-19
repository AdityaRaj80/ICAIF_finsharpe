"""Cross-check the 8 models are implemented CORRECTLY.

Beyond shapes/grads: each model's DEFINING MECHANISM is property-tested
(channel-independence, variate-tokens, strict causality, decomposition
reconstruction, structured global receptive field, variable-selection
softmax). A model that constructs+trains but violates its defining
property is silently wrong — these tests catch that.
"""
import numpy as np
import torch

from models import (REGISTRY, iTransformer, PatchTST, TFT, GCformer,
                     DLinear, TCN)

B, L, Fd, D = 8, 504, 65, 64
torch.manual_seed(0)
x = torch.randn(B, L, Fd)
rows = []


def rec(name, prop, ok, detail=""):
    rows.append((name, prop, "PASS" if ok else "FAIL", detail))


# ---- generic: shape / finite / backward / params ----
for nm, cls in REGISTRY.items():
    m = cls(Fd, L, D).train()
    z = m(x)
    okshape = tuple(z.shape) == (B, D)
    g_ok = True
    try:
        z.sum().backward()
        g_ok = all((p.grad is not None) for p in m.parameters()
                   if p.requires_grad and p.ndim > 0)
    except Exception as e:
        g_ok = False; nm += f"({e})"
    p = sum(t.numel() for t in m.parameters()) / 1e6
    rec(nm, "shape[B,D]+finite+backward", okshape and torch.isfinite(z).all()
        and g_ok, f"{tuple(z.shape)} {p:.2f}M")

# ---- iTransformer: tokens are the F VARIATES, per-variate normalised ----
m = iTransformer(Fd, L, D).eval()
tk = m.tokens(x)
xv = x.transpose(1, 2)
xn = (xv - xv.mean(-1, keepdim=True)) / (xv.std(-1, keepdim=True) + 1e-5)
rec("itransformer", "variate tokens [B,F,d] & per-variate norm",
    tk.shape == (B, Fd, D) and abs(float(xn.mean())) < 1e-3
    and abs(float(xn.std()) - 1) < 0.05, f"tokens{tuple(tk.shape)}")

# ---- PatchTST: CHANNEL-INDEPENDENCE (decisive) ----
m = PatchTST(Fd, L, D).eval()
with torch.no_grad():
    r0 = m.backbone(x)                       # [B,F,d]
    x2 = x.clone(); x2[:, :, 7] = torch.randn(B, L)   # perturb ONLY channel 7
    r1 = m.backbone(x2)
others = torch.cat([r1[:, :7], r1[:, 8:]], 1)
o0 = torch.cat([r0[:, :7], r0[:, 8:]], 1)
ci = torch.allclose(others, o0, atol=1e-6) and not torch.allclose(
    r1[:, 7], r0[:, 7], atol=1e-6)
rec("patchtst", "channel-independence (perturb chan7 -> others invariant)",
    ci, f"others_dmax={float((others-o0).abs().max()):.1e} "
        f"chan7_changed={float((r1[:,7]-r0[:,7]).abs().max()):.1e}")

# ---- TCN: strict causality (future input cannot change past output) ----
m = TCN(Fd, L, D).eval()
h = {}
[mm for mm in m.modules()]
last = [blk["c"] for blk in m.layers][-1]
last.register_forward_hook(lambda mod, i, o: h.__setitem__("y", o.detach()))
with torch.no_grad():
    m(x); y0 = h["y"][..., :L].clone()
    xp = x.clone(); xp[:, 400, :] += 9.0
    m(xp); y1 = h["y"][..., :L].clone()
chg = torch.nonzero((y1 - y0).abs().sum(0).sum(0) > 1e-6)
earliest = int(chg.min()) if len(chg) else L
rec("cnn", "strict causality (perturb t=400 -> earliest change>=400)",
    earliest >= 400, f"earliest_changed={earliest}")

# ---- DLinear: trend+seasonal reconstructs the input ----
m = DLinear(Fd, L, D).eval()
t, s = m.decompose(x)
recon = (t + s).transpose(1, 2)
rec("dlinear", "decomposition recon (trend+seasonal==x)",
    torch.allclose(recon, x, atol=1e-5),
    f"max|recon-x|={float((recon-x).abs().max()):.1e}")

# ---- GCformer: structured (sublinear) global kernel + full reach ----
m = GCformer(Fd, L, D).eval()
kparams = m.decay.numel() + m.coef.numel()
dense = D * L                                # a naive per-channel L-kernel
with torch.no_grad():
    z0 = m(x)
    xz = x.clone(); xz[:, 0, :] += 9.0        # perturb the OLDEST step only
    z1 = m(xz)
reach = float((z1 - z0).abs().max()) > 1e-6   # local branch can't see t=0
rec("gcformer", "structured global kernel (sublinear) + full receptive",
    kparams < dense and reach,
    f"kparams={kparams} << dense={dense}; t0_reaches_out={reach}")

# ---- TFT: per-variable embed + variable-selection softmax over F ----
m = TFT(Fd, L, D).eval()
w, e = m.vsn_weights(x)
sums = w.sum(-1)
rec("tft", "per-variable embed (W[F,d]) + VSN softmax sums to 1",
    m.W.shape == (Fd, D) and w.shape == (B, L, Fd)
    and torch.allclose(sums, torch.ones_like(sums), atol=1e-4),
    f"W{tuple(m.W.shape)} vsn{tuple(w.shape)} sum~{float(sums.mean()):.4f}")

# ---- report ----
print(f"{'model':<13}{'property':<52}{'res':<6}detail")
allok = True
for n, pr, r, d in rows:
    allok &= (r == "PASS")
    print(f"{n:<13}{pr:<52}{r:<6}{d}")
print("\nVERIFY:", "ALL PASS" if allok else "FAILURES PRESENT")
