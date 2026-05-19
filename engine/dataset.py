"""Date-grouped, purged, H-spaced loader over panel/features.parquet.

Closes models-jury F2. A batch is a block of rebalance DATES (not random
(stock,window) pairs); per date it carries the full eligible cross-section.
Selected rebalance dates are subsampled every H trading days (NON-overlapping
H-day label windows == PREREG §4 schedule) and kept >= H from the split
boundary (purge+embargo). Feature window = 504 rows ending at the rebalance
date; targets = forward H-day return/vol (already in the panel, causal).
"""
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch

PANEL = r"D:\Study\FinSharpe\panel\features.parquet"
SEQ_LEN = 504
NONFEAT = {"symbol", "date", "split"}


def _feature_cols(schema_names):
    return [c for c in schema_names
            if c not in NONFEAT and not c.startswith(("fwd_ret_", "fwd_vol_"))]


def load_panel(split, H, symbols=None):
    d = ds.dataset(PANEL, format="parquet")
    feat = _feature_cols(d.schema.names)
    cols = ["symbol", "date", "split", f"fwd_ret_{H}", f"fwd_vol_{H}"] + feat
    flt = ds.field("split") == split
    if symbols is not None:
        flt = flt & ds.field("symbol").isin(list(symbols))
    df = d.to_table(columns=cols, filter=flt).to_pandas()
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    return df, feat


class DateGroupedLoader:
    def __init__(self, split, H, symbols=None, min_dates=16,
                 max_stocks=None, seed=0):
        self.H = H
        df, self.feat = load_panel(split, H, symbols)
        # need the 504-lookback: pull the FULL history per symbol (any split)
        full = ds.dataset(PANEL, format="parquet").to_table(
            columns=["symbol", "date"] + self.feat,
            filter=(ds.field("symbol").isin(list(df.symbol.unique())))
        ).to_pandas().sort_values(["symbol", "date"]).reset_index(drop=True)
        self.X, self.dates_by_sym = {}, {}
        for s, g in full.groupby("symbol", sort=False):
            self.X[s] = g[self.feat].to_numpy(np.float32)
            self.dates_by_sym[s] = g["date"].to_numpy("datetime64[ns]")
        self.idx_of = {s: {d: i for i, d in enumerate(v)}
                       for s, v in self.dates_by_sym.items()}
        ry = df[[f"fwd_ret_{H}"]].to_numpy(np.float32).ravel()
        rv = df[[f"fwd_vol_{H}"]].to_numpy(np.float32).ravel()
        df = df.assign(yret=ry, yvol=rv)
        df = df.dropna(subset=["yret"])
        # eligible: full 504 window with no NaN
        elig = []
        for r in df.itertuples(index=False):
            s, dt = r.symbol, np.datetime64(r.date)
            j = self.idx_of[s].get(dt)
            if j is None or j < SEQ_LEN - 1:
                continue
            win = self.X[s][j - SEQ_LEN + 1:j + 1]
            if not np.isfinite(win).all():
                continue
            elig.append((dt, s, r.yret, r.yvol))
        e = pd.DataFrame(elig, columns=["date", "symbol", "y", "v"])
        # NON-overlapping H-spaced rebalance dates, >=H from boundary
        all_d = np.sort(e["date"].unique())
        keep = all_d[H:-H][::H] if len(all_d) > 2 * H else all_d[::H]
        self.e = e[e["date"].isin(keep)].reset_index(drop=True)
        self.sel_dates = np.sort(self.e["date"].unique())
        if max_stocks:
            self.e = (self.e.groupby("date", group_keys=False)
                      .apply(lambda x: x.sample(min(len(x), max_stocks),
                                                random_state=seed)))
        self.min_dates = min_dates
        self.code = {d: i for i, d in enumerate(self.sel_dates)}

    def batches(self):
        n = self.min_dates
        for i in range(0, len(self.sel_dates) - n + 1, n):
            blk = self.sel_dates[i:i + n]
            sub = self.e[self.e["date"].isin(blk)]
            xs, yr, yv, did = [], [], [], []
            for r in sub.itertuples(index=False):
                j = self.idx_of[r.symbol][np.datetime64(r.date)]
                xs.append(self.X[r.symbol][j - SEQ_LEN + 1:j + 1])
                yr.append(r.y); yv.append(r.v); did.append(self.code[r.date])
            yield {"x": torch.from_numpy(np.stack(xs)),
                   "y_ret": torch.tensor(yr, dtype=torch.float32),
                   "y_vol": torch.tensor(yv, dtype=torch.float32),
                   "date_id": torch.tensor(did, dtype=torch.long)}
