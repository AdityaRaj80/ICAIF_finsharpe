"""Phase-2 quicklook — early directional sanity check, NOT the H1 verdict.

Reads whatever {model}_{arm}_s{seed}_f{fold}.parquet files exist in
$P2_OUT/scores; per (backbone, arm) averages OOF scores across the
available seeds/folds; restricts to test-split rebalance dates; reports
the annualised hard-top-decile Sharpe and Spearman rank-IC vs y.

This is intentionally LITE: no DSR, no PBO, no Ridge requirement, no H1
accept rule. Its purpose is to answer "is the risk objective doing
anything distinguishable from the mse arm yet, on what we have so far?"
during the multi-day campaign. The full H1 verdict is engine/
phase2_aggregate.py, which requires every cell + ridge + tests for k.6.

Writes p2out/quicklook.md and prints the same table.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import DateGroupedLoader, load_panel
from backtest import _net_hard_port, _sharpe
from phase2 import POOL, H, MIN_DATES, OUT

BACKBONES = ["itransformer", "patchtst", "tft", "gcformer",
             "dlinear", "lstm", "rnn", "cnn"]
SC = os.path.join(OUT, "scores")


def _oof(model, arm, test_ids):
    fs = sorted(glob.glob(os.path.join(
        SC, f"{model}_{arm}_s*_f*.parquet")))
    if not fs:
        return None, 0, 0
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    g = (df.groupby(["date_id", "sym_id"], as_index=False)
           .agg(y=("y", "first"), score=("score", "mean")))
    g = g[g["date_id"].isin(test_ids)]
    return g, len(fs), len(g)


def main():
    L = DateGroupedLoader(POOL, H, min_dates=MIN_DATES)
    test_df, _ = load_panel("test", H)
    test_dates = set(pd.to_datetime(test_df["date"].unique())
                     .astype("datetime64[ns]"))
    test_ids = {L.code[d] for d in L.sel_dates if d in test_dates}
    print(f"[quicklook] test-split rebalance points = {len(test_ids)}")

    rows = []
    for b in BACKBONES + ["ridge"]:
        arms = ["mse", "risk"] if b != "ridge" else ["ridge"]
        for a in arms:
            g, n_files, n_rows = _oof(b, a, test_ids)
            if g is None or len(g) < 20:
                rows.append((b, a, n_files, n_rows, None, None))
                continue
            sc = g["score"].to_numpy()
            y = g["y"].to_numpy()
            did = g["date_id"].to_numpy()
            sid = g["sym_id"].to_numpy()
            port = _net_hard_port(sc, y, did, sid, cost=0.001)
            sr = _sharpe(port) if len(port) > 1 else None
            ic = spearmanr(sc, y).statistic
            rows.append((b, a, n_files, n_rows, sr, ic))

    hdr = ("| backbone | arm | cells | test_rows | annSharpe | rank-IC |")
    lines = ["# Phase-2 QUICKLOOK (early directional check, NOT H1 verdict)\n",
             f"_H={H}; test-split rebalance points={len(test_ids)}; "
             f"endpoint=hard top-decile equal-weight (cost=10bps); "
             f"PARTIAL — DSR/PBO/k.6 not computed; ridge may be missing._\n",
             hdr, "|" + "--|" * 6]
    for (b, a, n_files, n_rows, sr, ic) in rows:
        s = f"{sr:+.3f}" if sr is not None else "  .  "
        i = f"{ic:+.4f}" if ic is not None else "  .   "
        lines.append(f"| {b} | {a} | {n_files} | {n_rows} | {s} | {i} |")

    # If both arms present for any backbone, add the delta row
    risk_by_b = {b: r for r in rows for b in [r[0]] if r[1] == "risk"}
    mse_by_b = {b: r for r in rows for b in [r[0]] if r[1] == "mse"}
    paired = []
    for b in BACKBONES:
        if b in risk_by_b and b in mse_by_b:
            r, m = risk_by_b[b], mse_by_b[b]
            if r[4] is not None and m[4] is not None:
                paired.append((b, r[4] - m[4], (r[5] or 0) - (m[5] or 0)))
    if paired:
        lines += ["", "### Paired (risk − mse), partial:",
                  "| backbone | dSharpe_vs_mse | dRankIC_vs_mse |",
                  "|--|--|--|"]
        for (b, ds, di) in paired:
            lines.append(f"| {b} | {ds:+.3f} | {di:+.4f} |")
    lines += ["", "**WARNING**: this is a DIRECTIONAL CHECK with partial "
              "data. It is NOT the H1 verdict. Per PREREG §12 k.6, H1 "
              "requires the full eval, Ridge comparator, DSR≥0.95 vs both "
              "MSE and Ridge, PBO≤0.5, and is computed only by "
              "engine/phase2_aggregate.py on the complete 1275-cell grid."]

    out_path = os.path.join(OUT, "quicklook.md")
    open(out_path, "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[quicklook] wrote {out_path}")


if __name__ == "__main__":
    main()
