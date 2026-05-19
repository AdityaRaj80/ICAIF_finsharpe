"""Decision analysis: load price+news profiles, print distributions that
determine the modeling window and (data-driven) universe criteria."""
import numpy as np
import pandas as pd

px = pd.read_csv(r"D:\Study\FinSharpe\universe\price_profile.csv")
nw = pd.read_csv(r"D:\Study\FinSharpe\universe\news_profile.csv")
yrs = [f"y{y}" for y in range(2009, 2026)]

print("=" * 70)
print("NEWS: total articles per calendar year (all symbols)")
tot = nw[yrs].sum().astype(int)
for y, v in tot.items():
    print(f"  {y}: {v:>12,}")
print(f"  allext_total={int(nw.allext_count.sum()):,}  "
      f"nasdaq_total={int(nw.nasdaq_count.sum()):,}")

print("=" * 70)
print("NEWS: last_news date distribution (how far does news reach?)")
nw["last_y"] = nw["last_news"].str.slice(0, 4)
print(nw["last_y"].value_counts().sort_index().to_string())
print("  symbols with >=1 article in y2021..y2025:")
for y in ["y2021", "y2022", "y2023", "y2024", "y2025"]:
    print(f"    {y}: {(nw[y] > 0).sum():,} symbols, {int(nw[y].sum()):,} articles")

print("=" * 70)
print("PRICE: history start / end coverage")
px["first_y"] = px["first_date"].str.slice(0, 4)
px["last_d"] = pd.to_datetime(px["last_date"], errors="coerce")
print("  last_date year distribution:")
print(px["first_date"].notna().groupby(px["last_d"].dt.year).count().to_string())
for cut in ["2023-12-20", "2023-06-30", "2022-12-31", "2021-12-31", "2020-12-31"]:
    print(f"  last_date >= {cut}: {(px['last_d'] >= pd.Timestamp(cut)).sum():,}")
for cut in ["2009-01-01", "2010-01-01", "2011-01-01", "2012-01-01", "2013-01-01"]:
    px["first_d"] = pd.to_datetime(px["first_date"], errors="coerce")
    print(f"  first_date <= {cut}: {(px['first_d'] <= pd.Timestamp(cut)).sum():,}")

print("=" * 70)
print("PRICE: liquidity (med_dollar_vol_recent252) percentiles")
dv = px["med_dollar_vol_recent252"].replace(0, np.nan).dropna()
for p in [10, 25, 50, 60, 70, 75, 80, 90, 95]:
    print(f"  p{p}: ${np.percentile(dv, p):,.0f}")

print("=" * 70)
print("NEWS: total-article count percentiles (symbols with news)")
for p in [25, 50, 60, 70, 75, 80, 90, 95]:
    print(f"  p{p}: {np.percentile(nw['total'], p):,.0f}")

print("=" * 70)
print("JOINT screen sweep (price-complete x liquid x news-rich)")
px["first_d"] = pd.to_datetime(px["first_date"], errors="coerce")
m = px.merge(nw, on="ticker", how="left").fillna({"total": 0})
for fc, lc in [("2011-01-01", "2023-12-20"), ("2012-01-01", "2023-12-20"),
               ("2013-01-01", "2023-12-20")]:
    base = ((m["first_d"] <= pd.Timestamp(fc)) & (m["last_d"] >= pd.Timestamp(lc)))
    for dvf in [1e6, 5e6, 1e7]:
        for nf in [250, 500, 1000]:
            n = (base & (m["med_dollar_vol_recent252"] >= dvf)
                 & (m["total"] >= nf)).sum()
            print(f"  hist<={fc} end>={lc} dv>=${dvf:,.0f} news>={nf}: {n:,}")
print("=" * 70)
print("News in TEST years for the price-complete+liquid base "
      "(hist<=2012, end>=2023-12-20, dv>=$5M):")
base = ((m["first_d"] <= pd.Timestamp("2012-01-01"))
        & (m["last_d"] >= pd.Timestamp("2023-12-20"))
        & (m["med_dollar_vol_recent252"] >= 5e6))
b = m[base]
for y in ["y2019", "y2020", "y2021", "y2022", "y2023"]:
    print(f"  {y}: symbols with >0 = {(b[y] > 0).sum():,} / {len(b):,} ; "
          f"median articles among them = "
          f"{int(b.loc[b[y] > 0, y].median()) if (b[y] > 0).any() else 0}")
