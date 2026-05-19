"""Model assembly + smoke test.

build(name, arm) -> nn.Module(encoder + head). Run `python engine/registry.py`
to shape-check all 8 backbones on a synthetic DATE-GROUPED batch
([B,504,65]) for both arms — a cheap pre-training correctness gate.
"""
import torch
import torch.nn as nn

from models import REGISTRY
from heads import MSEReturnHead, RiskAwareHead, composite_risk_loss

SEQ_LEN, N_FEAT, D_MODEL = 504, 65, 64   # smoke default; real sizes set at train


class Model(nn.Module):
    def __init__(self, name, arm, n_feat=N_FEAT, seq_len=SEQ_LEN,
                 d_model=D_MODEL):
        super().__init__()
        if name not in REGISTRY:
            raise KeyError(f"{name} not in {list(REGISTRY)}")
        self.name, self.arm = name, arm
        self.enc = REGISTRY[name](n_feat, seq_len, d_model)
        self.head = (MSEReturnHead(d_model) if arm == "mse"
                     else RiskAwareHead(d_model))

    def forward(self, x):
        return self.enc(x)

    def loss(self, batch):
        z = self.enc(batch["x"])
        if self.arm == "mse":
            return self.head.loss(z, batch)        # already (loss, {})
        return composite_risk_loss(z, self.head, batch)


def build(name, arm):
    return Model(name, arm)


if __name__ == "__main__":
    torch.manual_seed(0)
    # synthetic DATE-GROUPED batch: 4 dates x 8 stocks (light smoke;
    # authoritative correctness = verify_models.py)
    n_dates, per = 4, 8
    N = n_dates * per
    batch = {
        "x": torch.randn(N, SEQ_LEN, N_FEAT),
        "y_ret": torch.randn(N) * 0.02,
        "y_vol": torch.rand(N) * 0.03,
        "date_id": torch.arange(n_dates).repeat_interleave(per),
    }
    print(f"{'model':<13}{'arm':<6}{'z.shape':<14}{'loss':>10}  params")
    ok = True
    for name in REGISTRY:
        for arm in ("mse", "risk"):
            try:
                m = build(name, arm)
                z = m(batch["x"])
                assert z.shape == (N, D_MODEL), f"z {z.shape}"
                L, parts = m.loss(batch)
                assert torch.isfinite(L), "non-finite loss"
                L.backward()
                p = sum(t.numel() for t in m.parameters()) / 1e6
                print(f"{name:<13}{arm:<6}{str(tuple(z.shape)):<14}"
                      f"{L.item():>10.4f}  {p:.2f}M"
                      + (f"  {parts}" if parts else ""))
            except Exception as e:
                ok = False
                print(f"{name:<13}{arm:<6}FAIL: {type(e).__name__}: {e}")
    print("\nSMOKE:", "ALL PASS" if ok else "FAILURES PRESENT")
