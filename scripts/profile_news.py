"""Stream both news CSVs and profile coverage per symbol.

Cheap, bounded-memory pass (counts + spans + per-year counts only; exact
per-day dedup is deferred to the small selected universe in verification).

Output: universe/news_profile.csv  one row per symbol:
  ticker, total, allext_count, nasdaq_count, first_news, last_news, y2009..y2025
"""
import sys, time
from collections import defaultdict
import pandas as pd

SOURCES = [
    # path, date_col, symbol_col, tag
    (r"D:\Study\FinSharpe\DATA\All_external.csv", "Date", "Stock_symbol", "allext"),
    (r"D:\Study\FinSharpe\DATA\Stock_news\nasdaq_exteral_data.csv", "Date", "Stock_symbol", "nasdaq"),
]
OUT = r"D:\Study\FinSharpe\universe\news_profile.csv"
CHUNK = 200_000

total = defaultdict(int)
by_src = defaultdict(lambda: defaultdict(int))      # tag -> sym -> count
first = {}                                          # sym -> 'YYYY-MM-DD'
last = {}                                           # sym -> 'YYYY-MM-DD'
by_year = defaultdict(int)                          # (sym, year) -> count

t0 = time.time()
for path, dc, sc, tag in SOURCES:
    print(f"[news] START {tag}: {path}", flush=True)
    n = 0
    reader = pd.read_csv(
        path, usecols=[dc, sc], dtype=str, chunksize=CHUNK,
        engine="c", on_bad_lines="skip", encoding_errors="replace",
    )
    for ci, ch in enumerate(reader):
        ch = ch.dropna()
        sym = ch[sc].astype(str).str.strip().str.upper()
        ds = ch[dc].astype(str).str.strip()
        d10 = ds.str.slice(0, 10)                    # YYYY-MM-DD (ISO -> lexsortable)
        yr = ds.str.slice(0, 4)
        ok = yr.str.fullmatch(r"\d{4}") & d10.str.fullmatch(r"\d{4}-\d{2}-\d{2}")
        sym, d10, yr = sym[ok], d10[ok], yr[ok]
        yi = yr.astype(int)
        ok2 = (yi >= 1990) & (yi <= 2026)
        sym, d10, yr = sym[ok2], d10[ok2], yr[ok2]

        vc = sym.value_counts()
        for s, c in vc.items():
            total[s] += int(c); by_src[tag][s] += int(c)
        gmin = d10.groupby(sym).min()
        gmax = d10.groupby(sym).max()
        for s, v in gmin.items():
            if s not in first or v < first[s]: first[s] = v
        for s, v in gmax.items():
            if s not in last or v > last[s]: last[s] = v
        for (s, y), c in sym.groupby([sym, yr]).size().items():
            by_year[(s, y)] += int(c)

        n += len(ch)
        if (ci + 1) % 25 == 0:
            print(f"[news] {tag} ~{n:,} rows  {time.time()-t0:,.0f}s  "
                  f"syms={len(total):,}", flush=True)
    print(f"[news] DONE {tag}: {n:,} rows  {time.time()-t0:,.0f}s", flush=True)

syms = sorted(total)
years = [str(y) for y in range(2009, 2026)]
recs = []
for s in syms:
    r = dict(ticker=s, total=total[s],
             allext_count=by_src["allext"].get(s, 0),
             nasdaq_count=by_src["nasdaq"].get(s, 0),
             first_news=first.get(s, ""), last_news=last.get(s, ""))
    for y in years:
        r[f"y{y}"] = by_year.get((s, y), 0)
    recs.append(r)

df = pd.DataFrame(recs)
df.to_csv(OUT, index=False)
print(f"[news] wrote {OUT}  ({len(df)} symbols)  total elapsed {time.time()-t0:,.0f}s",
      flush=True)
print(df["total"].describe().to_string(), flush=True)
