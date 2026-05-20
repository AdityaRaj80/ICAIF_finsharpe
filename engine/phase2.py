"""Phase-2 driver (Task 52). One file, three sub-commands:

  hpo   --model M --arm {mse,risk}        # 64-trial TPE, ONE selected
                                           # config / (model,arm)  (k.3)
  cell  --model M --arm A --seed S --fold F  # train selected config on
                                           # CPCV(6,2) path F train, predict
                                           # path F test -> scores parquet
  ridge --seed S --fold F                  # closed-form Ridge comparator
                                           # on IDENTICAL features (rebal-row)

Design (PREREG-faithful):
  pool = H-spaced non-overlapping rebalance dates over train+val+test
         (per-stock z already train-fit in the panel, §8).
  CPCV(6,2) -> 15 leakage-purged (train,test) paths over the pool.
  HPO is run ONCE per (model,arm) on nested_inner of a FIXED designated
  HPO path (path 0) -> one selected config; that config is then evaluated
  across the 15 paths x 5 seeds (seeds/paths = variance reduction of a
  fixed selected config -> NOT extra DSR trials; k.3 N=1152).
  Static tau=0.05 (registry.TAU; fully compliant per k.2 - validity is on
  the HARD endpoint scored by engine/backtest.py, never the surrogate).
  AMP + GradScaler + deterministic seed_all(seed). Encoder forward is
  gradient-checkpointed in CHUNKs so the full date-grouped cross-section
  fits 80GB at d_model=256 while keeping exact cross-sectional gradients.

Freeze: ensure_frozen() is called before ANY work -> first call stamps
PREREGISTRATION (k.1 binding). Scores are written to $P2_OUT/scores and
consumed by engine/phase2_aggregate.py -> backtest.score_h1 (hard
endpoint). Nothing here ever reads the soft surrogate for scoring.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from freeze import ensure_frozen
from determinism import seed_all
from dataset import DateGroupedLoader, cpcv_folds, nested_inner
from registry import Model, TAU
from heads import composite_risk_loss

POOL = ("train", "val", "test")
H = int(os.environ.get("P2_H", "5"))
# P2_SMOKE=1 ONLY for the local correctness smoke (never set by the
# sbatch): shrinks net/epochs/HPO/universe so the FULL pipeline + freeze
# routing can be exercised fast. Production keeps the PREREG §9 pins.
SMOKE = os.environ.get("P2_SMOKE") == "1"
D_MODEL = 32 if SMOKE else 256                 # PREREG §9 pin (256)
SEQ_LEN = 504
MIN_DATES = 4 if SMOKE else 16                 # PREREG §9 pin (16)
CHUNK = int(os.environ.get("P2_CHUNK", "256"))  # encoder micro-batch
N_HPO = 4 if SMOKE else 64                     # PREREG §9 / k.3 pin (64)
MAX_EPOCH = 2 if SMOKE else 50
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.environ.get("P2_OUT", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "p2out"))
SYMS_FILE = os.environ.get(
    "P2_SYMS", os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "universe", "tier1.txt"))


# ----------------------------------------------------------------------
# base pool loader (built once; folds are in-memory date subsets)
# ----------------------------------------------------------------------
_BASE = {}


def base_loader(symbols=None):
    key = "B"
    if key not in _BASE:
        if symbols is None and SMOKE:
            import pandas as _pd
            symbols = (_pd.read_csv(SYMS_FILE, header=None)[0]
                       .astype(str).str.strip().tolist()[:12])
        _BASE[key] = DateGroupedLoader(POOL, H, symbols=symbols,
                                       min_dates=MIN_DATES)
    return _BASE[key]


def _block_iter(L, dates, min_dates=MIN_DATES, shuffle=False, seed=0):
    """Yield date-grouped batches restricted to `dates`, reusing the base
    loader's tensors/codes (no parquet re-read). did/sym_id use the base
    loader's global codes so score_h1 alignment is stable."""
    d = np.sort(np.asarray(dates, dtype="datetime64[ns]"))
    sub = L.e[L.e["date"].isin(d)]
    order = np.arange(len(d))
    if shuffle:
        rng = np.random.default_rng(seed)
        # shuffle WHOLE blocks, never rows (keeps cross-sections intact)
        nb = max(1, len(d) // min_dates)
        bo = rng.permutation(nb)
    for bi in range(0, len(d) - min_dates + 1, min_dates):
        i = bi
        if shuffle:
            i = (bo[bi // min_dates] if bi // min_dates < len(bo)
                 else bi // min_dates) * min_dates
            if i + min_dates > len(d):
                continue
        blk = d[i:i + min_dates]
        s = sub[sub["date"].isin(blk)]
        if s.empty:
            continue
        xs, yr, yv, did, sid = [], [], [], [], []
        for r in s.itertuples(index=False):
            j = L.idx_of[r.symbol][np.datetime64(r.date)]
            xs.append(L.X[r.symbol][j - SEQ_LEN + 1:j + 1])
            yr.append(r.y); yv.append(r.v)
            did.append(L.code[r.date]); sid.append(L.sym_code[r.symbol])
        yield {"x": torch.from_numpy(np.stack(xs)),
               "y_ret": torch.tensor(yr, dtype=torch.float32),
               "y_vol": torch.tensor(yv, dtype=torch.float32),
               "date_id": torch.tensor(did, dtype=torch.long),
               "sym_id": torch.tensor(sid, dtype=torch.long)}


def _encode(enc, x, train):
    """Chunked encoder forward. In training, gradient-checkpoint each
    chunk -> activation memory O(CHUNK), exact grads recomputed in
    backward (deterministic with cudnn.deterministic + seed_all)."""
    zs = []
    for k in range(0, x.shape[0], CHUNK):
        xc = x[k:k + CHUNK]
        if train:
            zc = checkpoint(enc, xc, use_reentrant=False)
        else:
            zc = enc(xc)
        zs.append(zc)
    return torch.cat(zs, 0)


def _loss(model, batch, scaler_dev):
    z = _encode(model.enc, batch["x"], model.training)
    if model.arm == "mse":
        return F.mse_loss(model.head(z), batch["y_ret"])
    tot, _ = composite_risk_loss(z, model.head, batch, tau=TAU)
    return tot


def _val_metric(model, L, val_dates):
    """Selection / early-stop metric on val: risk -> annualised hard
    top-decile Sharpe of mu-scores; mse -> negative MSE."""
    from backtest import _net_hard_port, _sharpe
    model.eval()
    sc, ys, dd, ss = [], [], [], []
    with torch.no_grad():
        for b in _block_iter(L, val_dates):
            b = {k: v.to(DEV) for k, v in b.items()}
            z = _encode(model.enc, b["x"], False)
            out = model.head(z)
            mu = out[0] if isinstance(out, tuple) else out
            sc.append(mu.float().cpu().numpy())
            ys.append(b["y_ret"].cpu().numpy())
            dd.append(b["date_id"].cpu().numpy())
            ss.append(b["sym_id"].cpu().numpy())
    if not sc:
        return -1e9
    sc = np.concatenate(sc); ys = np.concatenate(ys)
    dd = np.concatenate(dd); ss = np.concatenate(ss)
    if model.arm == "mse":
        return -float(np.mean((sc - ys) ** 2))
    port = _net_hard_port(sc, ys, dd, ss, cost=0.0)
    return float(_sharpe(port)) if len(port) > 1 else -1e9


def _train(model, L, train_dates, hp, seed, max_epoch=MAX_EPOCH,
           trial=None):
    """Train with nested_inner early-stop on the TRAIN fold (HPO uses the
    same routine). `trial` (HPO only) enables Optuna median-pruning:
    pruning early-kills hopeless trials but the SAMPLED trial count stays
    N_HPO=64, so the DSR deflation N=1152 (k.3) is unchanged — disclosed.
    Returns best-val state restored into `model`."""
    import optuna
    inner_tr, inner_val = nested_inner(train_dates)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"],
                            weight_decay=hp["wd"])
    scaler = torch.amp.GradScaler(DEV, enabled=(DEV == "cuda"))
    best, best_state, bad = -1e18, None, 0
    for ep in range(max_epoch):
        model.train()
        for b in _block_iter(L, inner_tr, shuffle=True, seed=seed * 100 + ep):
            b = {k: v.to(DEV) for k, v in b.items()}
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(DEV, enabled=(DEV == "cuda")):
                loss = _loss(model, b, DEV)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp["clip"])
            scaler.step(opt); scaler.update()
        m = _val_metric(model, L, inner_val)
        if trial is not None:
            trial.report(m, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if m > best + 1e-9:
            best, bad = m, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= hp["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best


def _predict(model, L, test_dates):
    model.eval()
    rows = []
    with torch.no_grad():
        for b in _block_iter(L, test_dates):
            bd = {k: v.to(DEV) for k, v in b.items()}
            z = _encode(model.enc, bd["x"], False)
            out = model.head(z)
            mu = out[0] if isinstance(out, tuple) else out
            mu = mu.float().cpu().numpy()
            did = b["date_id"].numpy(); sid = b["sym_id"].numpy()
            y = b["y_ret"].numpy()
            for i in range(len(mu)):
                rows.append((int(did[i]), int(sid[i]),
                             float(y[i]), float(mu[i])))
    return pd.DataFrame(rows, columns=["date_id", "sym_id", "y", "score"])


# ----------------------------------------------------------------------
# Ridge comparator (closed form) on IDENTICAL features = rebalance-row x
# ----------------------------------------------------------------------
def _xrow(L, dates):
    sub = L.e[L.e["date"].isin(np.asarray(dates, "datetime64[ns]"))]
    X, y, did, sid = [], [], [], []
    for r in sub.itertuples(index=False):
        j = L.idx_of[r.symbol][np.datetime64(r.date)]
        X.append(L.X[r.symbol][j])                     # last (rebalance) row
        y.append(r.y); did.append(L.code[r.date])
        sid.append(L.sym_code[r.symbol])
    return (np.asarray(X, np.float64), np.asarray(y, np.float64),
            np.asarray(did), np.asarray(sid))


def _done(model, arm, seed, fold):
    return os.path.exists(os.path.join(
        OUT, "scores", f"{model}_{arm}_s{seed}_f{fold}.parquet"))


def run_ridge(seed, fold):
    if _done("ridge", "ridge", seed, fold):
        print(f"[p2] skip ridge s{seed} f{fold} (exists)", flush=True)
        return
    sel = os.path.join(OUT, "selected", "ridge_ridge.json")
    if not os.path.exists(sel):
        print(f"[p2] DEFER ridge s{seed} f{fold}: ridge HPO not done.",
              flush=True)
        return
    ensure_frozen("phase2-ridge")
    L = base_loader()
    folds = cpcv_folds(L.sel_dates)
    tr, te = folds[fold]
    alpha = json.load(open(sel))["alpha"]
    Xtr, ytr, _, _ = _xrow(L, tr)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr = (Xtr - mu) / sd
    p = Xtr.shape[1]
    W = np.linalg.solve(Xtr.T @ Xtr + alpha * np.eye(p), Xtr.T @ ytr)
    Xte, yte, did, sid = _xrow(L, te)
    sc = ((Xte - mu) / sd) @ W
    df = pd.DataFrame({"date_id": did, "sym_id": sid, "y": yte,
                       "score": sc})
    _save(df, "ridge", "ridge", seed, fold)


# ----------------------------------------------------------------------
def _save(df, model, arm, seed, fold):
    d = os.path.join(OUT, "scores")
    os.makedirs(d, exist_ok=True)
    fp = os.path.join(d, f"{model}_{arm}_s{seed}_f{fold}.parquet")
    df.to_parquet(fp, index=False)
    print(f"[p2] wrote {fp}  rows={len(df)}", flush=True)


def _space(trial, model):
    if model == "ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True)}
    return {"lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "wd": trial.suggest_float("wd", 1e-7, 1e-3, log=True),
            "drop": trial.suggest_categorical("drop", [0.0, 0.1, 0.2]),
            "clip": trial.suggest_categorical("clip", [0.5, 1.0, 5.0]),
            "patience": trial.suggest_categorical("patience", [4, 6, 10])}


def run_hpo(model, arm):
    sel = os.path.join(OUT, "selected", f"{model}_{arm}.json")
    if os.path.exists(sel):
        print(f"[p2] skip HPO {model}/{arm} (exists {sel})", flush=True)
        return
    ensure_frozen("phase2-hpo")
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    L = base_loader()
    folds = cpcv_folds(L.sel_dates)
    tr0, _ = folds[0]                              # FIXED designated HPO path
    os.makedirs(os.path.join(OUT, "selected"), exist_ok=True)

    if model == "ridge":
        in_tr, in_val = nested_inner(tr0)
        Xtr, ytr, _, _ = _xrow(L, in_tr)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        Xtr = (Xtr - mu) / sd
        Xv, yv, dv, _ = _xrow(L, in_val)
        Xv = (Xv - mu) / sd
        p = Xtr.shape[1]

        def obj(t):
            a = _space(t, "ridge")["alpha"]
            W = np.linalg.solve(Xtr.T @ Xtr + a * np.eye(p), Xtr.T @ ytr)
            return float(np.mean((Xv @ W - yv) ** 2))           # min MSE
        st = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=0))
        st.optimize(obj, n_trials=N_HPO)
        best = st.best_params
    else:
        def obj(t):
            seed_all(0)
            hp = _space(t, model)
            m = Model(model, arm, n_feat=len(L.feat), seq_len=SEQ_LEN,
                      d_model=D_MODEL).to(DEV)
            for mod in m.modules():
                if isinstance(mod, torch.nn.Dropout):
                    mod.p = hp["drop"]
            v = _train(m, L, tr0, hp, seed=0,
                       max_epoch=min(25, MAX_EPOCH),
                       trial=t)                                # HPO budget
            return -v                                          # maximise v
        st = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=0),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=8,
                                               n_warmup_steps=5))
        st.optimize(obj, n_trials=N_HPO, gc_after_trial=True)
        best = st.best_params
        best.setdefault("patience", best.get("patience", 6))
    fp = os.path.join(OUT, "selected", f"{model}_{arm}.json")
    json.dump(best, open(fp, "w"), indent=2)
    print(f"[p2] HPO done {model}/{arm} -> {best}  ({fp})", flush=True)


def run_cell(model, arm, seed, fold):
    if _done(model, arm, seed, fold):
        print(f"[p2] skip {model}/{arm} s{seed} f{fold} (exists)",
              flush=True)
        return
    sel = os.path.join(OUT, "selected", f"{model}_{arm}.json")
    if not os.path.exists(sel):
        print(f"[p2] DEFER {model}/{arm} s{seed} f{fold}: HPO not done "
              f"yet ({sel} missing). Re-run phase2_submit.sh to retry.",
              flush=True)
        return                                          # exit 0, restart-safe
    ensure_frozen("phase2-cell")
    seed_all(seed)
    L = base_loader()
    folds = cpcv_folds(L.sel_dates)
    tr, te = folds[fold]
    hp = json.load(open(sel))
    hp = {"lr": hp.get("lr", 1e-3), "wd": hp.get("wd", 1e-5),
          "drop": hp.get("drop", 0.1), "clip": hp.get("clip", 1.0),
          "patience": hp.get("patience", 6)}
    m = Model(model, arm, n_feat=len(L.feat), seq_len=SEQ_LEN,
              d_model=D_MODEL).to(DEV)
    for mod in m.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = hp["drop"]
    t0 = time.time()
    v = _train(m, L, tr, hp, seed=seed)
    df = _predict(m, L, te)
    _save(df, model, arm, seed, fold)
    print(f"[p2] cell {model}/{arm} s{seed} f{fold} best_val={v:.4f} "
          f"{time.time() - t0:,.0f}s", flush=True)


BACKBONES = ["itransformer", "patchtst", "tft", "gcformer",
             "dlinear", "lstm", "rnn", "cnn"]
ARMS = ["mse", "risk"]
N_SEED, N_FOLD = 5, 15
N_HPO_TASKS = len(BACKBONES) * len(ARMS) + 1            # +1 ridge = 17
N_EVAL_DEEP = len(BACKBONES) * len(ARMS) * N_SEED * N_FOLD   # 1200
N_EVAL_TASKS = N_EVAL_DEEP + N_SEED * N_FOLD            # +75 ridge = 1275


def dispatch(phase, idx):
    """Single source of truth mapping SLURM_ARRAY_TASK_ID -> work unit."""
    if phase == "hpo":
        if idx == N_HPO_TASKS - 1:
            return run_hpo("ridge", "ridge")
        return run_hpo(BACKBONES[idx // 2], ARMS[idx % 2])
    if idx >= N_EVAL_DEEP:                              # ridge cells
        r = idx - N_EVAL_DEEP
        return run_ridge(seed=r // N_FOLD, fold=r % N_FOLD)
    fold = idx % N_FOLD
    t = idx // N_FOLD
    seed = t % N_SEED
    t //= N_SEED
    arm = ARMS[t % len(ARMS)]
    model = BACKBONES[t // len(ARMS)]
    return run_cell(model, arm, seed, fold)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hpo"); h.add_argument("--model", required=True)
    h.add_argument("--arm", required=True)
    c = sub.add_parser("cell"); c.add_argument("--model", required=True)
    c.add_argument("--arm", required=True)
    c.add_argument("--seed", type=int, required=True)
    c.add_argument("--fold", type=int, required=True)
    r = sub.add_parser("ridge"); r.add_argument("--seed", type=int,
                                                required=True)
    r.add_argument("--fold", type=int, required=True)
    d = sub.add_parser("dispatch")
    d.add_argument("--phase", required=True, choices=["hpo", "eval"])
    d.add_argument("--idx", type=int, required=True)
    a = ap.parse_args()
    if a.cmd == "hpo":
        run_hpo(a.model, a.arm)
    elif a.cmd == "cell":
        run_cell(a.model, a.arm, a.seed, a.fold)
    elif a.cmd == "ridge":
        run_ridge(a.seed, a.fold)
    else:
        dispatch(a.phase, a.idx)
