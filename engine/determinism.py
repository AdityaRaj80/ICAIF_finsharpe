"""Determinism harness (re-jury PREREG §7/§8).

seed_all() pins every RNG + forces deterministic kernels so each of the
5 seeds {0..4} is bit-reproducible. Run `python engine/determinism.py`
for the 2-run bit-identical proof (required before any Phase-2 training).
"""
import os
import random

import numpy as np
import torch


def seed_all(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    # cuBLAS determinism for matmul on CUDA (must be set before CUDA init)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        torch.use_deterministic_algorithms(True, warn_only=True)
    g = torch.Generator()
    g.manual_seed(seed)
    return g                                   # pass to DataLoader(generator=g)


def _one_run(seed):
    seed_all(seed)
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from models import iTransformer
    from heads import RiskAwareHead, composite_risk_loss
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    x = torch.randn(64, 504, 65, device=dev)
    batch = {"x": x,
             "y_ret": torch.randn(64, device=dev) * 0.02,
             "y_vol": torch.rand(64, device=dev) * 0.03,
             "date_id": torch.arange(8, device=dev).repeat_interleave(8)}
    enc = iTransformer(65, 504, 64).to(dev)
    head = RiskAwareHead(64).to(dev)
    z = enc(batch["x"])
    loss, _ = composite_risk_loss(z, head, batch, tau=0.05)
    loss.backward()
    gnorm = torch.cat([p.grad.flatten() for p in enc.parameters()
                       if p.grad is not None]).norm()
    return float(loss), float(gnorm)


if __name__ == "__main__":
    a = _one_run(0)
    b = _one_run(0)            # same seed -> must be bit-identical
    c = _one_run(1)            # different seed -> must differ
    ident = (a == b)
    diff = (a != c)
    print(f"seed0 run1 = {a}")
    print(f"seed0 run2 = {b}   bit-identical={ident}")
    print(f"seed1      = {c}   differs-from-seed0={diff}")
    print("DETERMINISM:", "PASS" if (ident and diff) else "FAIL")
