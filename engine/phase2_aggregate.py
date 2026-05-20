"""Phase-2 aggregator -> the H1 verdict (PREREG §12 k.6 / PAPER_PLAN H1).

Consumes $P2_OUT/scores/{model}_{arm}_s{seed}_f{fold}.parquet written by
engine/phase2.py and produces the single primary table.

k.3 is explicit: SEEDS and CPCV PATHS are variance reduction of a *fixed
selected config* (not selectable strategies). So the point estimate
averages each (date_id,sym_id) score over the CPCV paths where it is
out-of-sample AND over the 5 seeds -> ONE risk / ONE mse / ONE ridge
score per (date,stock) per backbone. The H1 return series is then
restricted to the TEST split (2021-2023; PREREG §1 n_eff~150) and scored
EXCLUSIVELY by engine/backtest.py `score_h1` -> `hard_top_decile_returns`
(the hard endpoint; never the soft surrogate). Per-seed Sharpe dispersion
is reported descriptively (appendix), never as an extra DSR trial.

Writes bench/phase2_results.md (primary table + per-backbone H1_ACCEPT +
overall verdict) and p2out/phase2_raw.json.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import DateGroupedLoader, cpcv_folds, load_panel
from backtest import score_h1
from phase2 import POOL, H, MIN_DATES, OUT

BACKBONES = ["itransformer", "patchtst", "tft", "gcformer",
             "dlinear", "lstm", "rnn", "cnn"]
SC = os.path.join(OUT, "scores")


def _oof(model, arm):
    """Mean score per (date_id,sym_id) over all available seed/fold
    files (CPCV-OOF + seed averaging; k.3 variance reduction)."""
    fs = sorted(glob.glob(os.path.join(SC, f"{model}_{arm}_s*_f*.parquet")))
    if not fs:
        return None
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    g = (df.groupby(["date_id", "sym_id"], as_index=False)
           .agg(y=("y", "first"), score=("score", "mean")))
    return g


def _per_seed_sharpe(model, arm, test_ids):
    """Descriptive only: hard-endpoint Sharpe per seed (appendix
    dispersion). Never enters DSR/PBO/k.6."""
    from backtest import _net_hard_port, _sharpe
    out = []
    for s in range(5):
        fs = sorted(glob.glob(os.path.join(
            SC, f"{model}_{arm}_s{s}_f*.parquet")))
        if not fs:
            continue
        df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
        df = df[df["date_id"].isin(test_ids)]
        if df.empty:
            continue
        g = (df.groupby(["date_id", "sym_id"], as_index=False)
               .agg(y=("y", "first"), score=("score", "mean")))
        p = _net_hard_port(g["score"].to_numpy(), g["y"].to_numpy(),
                           g["date_id"].to_numpy(), g["sym_id"].to_numpy())
        if len(p) > 1:
            out.append(_sharpe(p))
    return out


def main():
    L = DateGroupedLoader(POOL, H, min_dates=MIN_DATES)
    _ = cpcv_folds(L.sel_dates)                       # validate fold grid
    test_df, _f = load_panel("test", H)
    test_dates = set(pd.to_datetime(test_df["date"].unique())
                     .astype("datetime64[ns]"))
    test_ids = {L.code[d] for d in L.sel_dates if d in test_dates}
    print(f"[agg] test-split rebalance ids = {len(test_ids)} "
          f"(PREREG §1 n_eff target ~150)")

    ridge = _oof("ridge", "ridge")
    if ridge is None:
        sys.exit("[agg] no ridge scores yet — run ridge cells first.")
    ridge = ridge[ridge["date_id"].isin(test_ids)]

    rows, raw = [], {}
    for b in BACKBONES:
        r = _oof(b, "risk")
        m = _oof(b, "mse")
        if r is None or m is None:
            print(f"[agg] SKIP {b}: missing risk/mse scores")
            continue
        key = ["date_id", "sym_id"]
        j = (r.rename(columns={"score": "s_risk"})
             .merge(m.rename(columns={"score": "s_mse"})[key + ["s_mse"]],
                    on=key)
             .merge(ridge.rename(columns={"score": "s_rdg"})[
                 key + ["s_rdg"]], on=key))
        j = j[j["date_id"].isin(test_ids)]
        if len(j) < 50:
            print(f"[agg] SKIP {b}: only {len(j)} aligned test rows")
            continue
        res = score_h1(j["s_risk"].to_numpy(), j["s_mse"].to_numpy(),
                        j["s_rdg"].to_numpy(), j["y"].to_numpy(),
                        j["date_id"].to_numpy(), j["sym_id"].to_numpy())
        res["seed_sharpe_risk"] = _per_seed_sharpe(b, "risk", test_ids)
        raw[b] = res
        rows.append((b, res["sharpe_risk"], res["sharpe_mse"],
                     res["sharpe_ridge"], res["dSharpe_vs_mse"],
                     res["dSharpe_vs_ridge"], res["dRankIC_vs_mse"],
                     res["p_vs_mse"], res["p_vs_ridge"], res["PBO"],
                     res["H1_ACCEPT"]))

    os.makedirs(OUT, exist_ok=True)
    json.dump(raw, open(os.path.join(OUT, "phase2_raw.json"), "w"),
              indent=2, default=float)

    hdr = ("| backbone | SR_risk | SR_mse | SR_rdg | dSR_vs_mse | "
           "dSR_vs_rdg | dRankIC | p_vs_mse | p_vs_rdg | PBO | H1 |")
    lines = ["# Phase-2 H1 results (hard endpoint; PREREG §12 k.6)\n",
             f"_H={H}; test-split rebalance points={len(test_ids)}; "
             f"endpoint=hard_top_decile_returns; seeds&CPCV averaged "
             f"(k.3 variance reduction); DSR N=1152; PBO=CSCV S=10/252._\n",
             hdr, "|" + "--|" * 11]
    any_accept = False
    for (b, sr, sm, sg, dm, dg, di, pm, pg, pbo, acc) in rows:
        any_accept |= bool(acc)
        lines.append(f"| {b} | {sr:.3f} | {sm:.3f} | {sg:.3f} | {dm:.3f} "
                     f"| {dg:.3f} | {di:.4f} | {pm:.3f} | {pg:.3f} | "
                     f"{pbo:.3f} | {'ACCEPT' if acc else 'null'} |")
    verdict = ("H1 ACCEPTED for >=1 backbone (see table)" if any_accept
               else "H1 NULL for ALL 8 backbones — the pre-registered "
               "honest-negative outcome (k.1: reported as a failure, "
               "NOT amended)")
    lines += ["", f"**VERDICT: {verdict}**", "",
              "_Per-seed Sharpe dispersion (risk arm) is descriptive "
              "appendix only; excluded from DSR/PBO/k.6._"]
    rep = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "bench", "phase2_results.md")
    open(rep, "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[agg] wrote {rep}")


if __name__ == "__main__":
    main()
