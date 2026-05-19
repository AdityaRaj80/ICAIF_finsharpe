"""CI GUARD (re-jury-5 FATAL-2): PREREGISTRATION.md §9 vs the actual code.

Imports the engine and FAILS (exit 1) if any frozen constant drifts from
the §9-declared value. Run before the freeze stamp and in CI; prevents the
stale-§9 class of defect (the prior §9 pinned an `α_pos 10` that no longer
existed). The CANON below MUST mirror PREREGISTRATION.md §9 verbatim.
"""
import inspect
import sys

import heads
import registry
import dataset
from models import (DLinear, LSTMEnc, RNNEnc, TCN, iTransformer, PatchTST,
                    GCformer, TFT)


def dft(fn, name):
    return inspect.signature(fn).parameters[name].default


CANON = {
    # objective / loss (heads.py)
    "heads.FRAC": (heads.FRAC, 0.10),
    "heads.MIN_DATES": (heads.MIN_DATES, 16),
    "heads.EPS": (heads.EPS, 1e-8),
    "composite.a": (dft(heads.composite_risk_loss, "a"), 0.7),
    "composite.g": (dft(heads.composite_risk_loss, "g"), 0.5),
    "composite.b": (dft(heads.composite_risk_loss, "b"), 0.5),
    "heads.COST": (heads.COST, 0.001),
    "composite.cost": (dft(heads.composite_risk_loss, "cost"), heads.COST),
    # sequence / registry
    "registry.SEQ_LEN": (registry.SEQ_LEN, 504),
    "registry.N_FEAT": (registry.N_FEAT, 65),
    "registry.TAU": (registry.TAU, 0.05),
    "dataset.SEQ_LEN": (dataset.SEQ_LEN, 504),
    # backbone training capacity = each model's OWN default d_model (256)
    "d_model(DLinear)": (dft(DLinear.__init__, "d_model"), 256),
    "d_model(PatchTST)": (dft(PatchTST.__init__, "d_model"), 256),
    "d_model(iTransformer)": (dft(iTransformer.__init__, "d_model"), 256),
    "d_model(GCformer)": (dft(GCformer.__init__, "d_model"), 256),
    "d_model(TFT)": (dft(TFT.__init__, "d_model"), 256),
    # per-backbone constants (research-grade)
    "DLinear.kernel": (dft(DLinear.__init__, "kernel"), 25),
    "LSTMEnc.layers": (dft(LSTMEnc.__init__, "layers"), 3),
    "RNNEnc.layers": (dft(RNNEnc.__init__, "layers"), 3),
    "iTransformer.heads": (dft(iTransformer.__init__, "heads"), 8),
    "iTransformer.layers": (dft(iTransformer.__init__, "layers"), 4),
    "PatchTST.patch": (dft(PatchTST.__init__, "patch"), 16),
    "PatchTST.stride": (dft(PatchTST.__init__, "stride"), 8),
    "PatchTST.heads": (dft(PatchTST.__init__, "heads"), 8),
    "PatchTST.layers": (dft(PatchTST.__init__, "layers"), 4),
    "GCformer.K": (dft(GCformer.__init__, "K"), 24),
    "GCformer.local": (dft(GCformer.__init__, "local"), 128),
    "TFT.heads": (dft(TFT.__init__, "heads"), 4),
    # CPCV (dataset.py)
    "cpcv.n_groups": (dft(dataset.cpcv_folds, "n_groups"), 6),
    "cpcv.k": (dft(dataset.cpcv_folds, "k"), 2),
    "cpcv.purge": (dft(dataset.cpcv_folds, "purge"), 1),
    "nested.n_groups": (dft(dataset.nested_inner, "n_groups"), 5),
    "nested.val_groups": (dft(dataset.nested_inner, "val_groups"), 1),
    "nested.purge": (dft(dataset.nested_inner, "purge"), 1),
}

bad = [(k, c, e) for k, (c, e) in CANON.items()
       if (abs(c - e) > 1e-12 if isinstance(e, float) else c != e)]
for k, (c, e) in CANON.items():
    print(f"  {'OK ' if (k not in [b[0] for b in bad]) else 'BAD'} "
          f"{k:<26} code={c!r:<10} §9={e!r}")
if bad:
    print(f"\nDRIFT: {len(bad)} constant(s) differ from PREREG §9 -> "
          f"FIX §9 OR THE CODE BEFORE FREEZE")
    for k, c, e in bad:
        print(f"  {k}: code={c!r} != §9={e!r}")
    sys.exit(1)
print(f"\nPREREG §9 CONSISTENT with code ({len(CANON)} constants checked).")
