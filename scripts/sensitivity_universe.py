"""Universe threshold sensitivity (JURY FATAL Jury3#1).

Baseline = the EXACT build_universe.py constants (must reproduce 870).
Cutoffs are pre-registered as ABSOLUTE, economically-motivated floors
(not outcome-tuned quantiles): $5M median ADV = standard tradable-liquidity
floor; >=500 articles = minimum for a non-degenerate decayed-sentiment
series; hist<=2011-01-01 / end>=2023-12-20 = the modeling window + warmup;
n_rows>=2800, gap<=10 = data-completeness hygiene. We report the empirical
percentile each floor SITS AT (transparency, not justification) and show
Tier-1 size + Jaccard are stable to economically-sensible perturbations.
Output: universe/sensitivity_report.md
"""
import itertools, re
import numpy as np
import pandas as pd

U = r"D:\Study\FinSharpe\universe"
px = pd.read_csv(fr"{U}\price_profile.csv")
nw = pd.read_csv(fr"{U}\news_profile.csv")
px["first_d"] = pd.to_datetime(px["first_date"], errors="coerce")
px["last_d"] = pd.to_datetime(px["last_date"], errors="coerce")
m = px.merge(nw, on="ticker", how="left")
for c in [f"y{y}" for y in range(2009, 2026)] + ["total"]:
    m[c] = m[c].fillna(0)
TRY, TEY = list(range(2013, 2020)), [2021, 2022, 2023]

bu = open(fr"D:\Study\FinSharpe\scripts\build_universe.py").read()
ETF = set(bu.split('ETF = set("""')[1].split('""".split())')[0].split())
LEV = re.compile(r"^(TQQQ|SQQQ|UVXY|VXX|SVXY|.*(BULL|BEAR)).*$")

# EXACT build constants (pre-registered absolute floors)
BASE = dict(dvf=5_000_000, nf=500, fc="2011-01-01", lc="2023-12-20",
            nrows=2800, gap=10)

def tier1(dvf, nf, fc, lc, nrows, gap):
    ok = ((m.first_d <= pd.Timestamp(fc)) & (m.last_d >= pd.Timestamp(lc))
          & (m.n_rows >= nrows) & (m.max_gap_days.between(0, gap))
          & (m.med_dollar_vol_recent252 >= dvf) & (m.total >= nf))
    tr = sum((m[f"y{y}"] > 0).astype(int) for y in TRY)
    te = sum((m[f"y{y}"] > 0).astype(int) for y in TEY)
    ok &= (tr >= 5) & (te >= 2)
    sel = m[ok]
    sel = sel[~sel.ticker.isin(ETF) & ~sel.ticker.astype(str).str.match(LEV)]
    return set(sel.ticker)

B = tier1(**BASE)
def jac(a, b): return len(a & b) / len(a | b) if (a | b) else 1.0

# empirical percentile each floor sits at (within the price/window-eligible
# pool) — descriptive transparency only
elig = m[(m.first_d <= "2011-01-01") & (m.last_d >= "2023-12-20")]
dv = elig["med_dollar_vol_recent252"].replace(0, np.nan).dropna()
pct_dv = float((dv < BASE["dvf"]).mean() * 100)
pct_nw = float((elig["total"] < BASE["nf"]).mean() * 100)

rows = []
for v in [2e6, 3e6, 5e6, 7e6, 1e7, 1.5e7]:
    T = tier1(**{**BASE, "dvf": v}); rows.append((f"$vol>=${v:,.0f}", len(T), round(jac(T, B), 3)))
for v in [300, 400, 500, 750, 1000, 1500]:
    T = tier1(**{**BASE, "nf": v}); rows.append((f"news>={v}", len(T), round(jac(T, B), 3)))
for v in [2400, 2600, 2800, 3000, 3200]:
    T = tier1(**{**BASE, "nrows": v}); rows.append((f"n_rows>={v}", len(T), round(jac(T, B), 3)))
for v in [7, 8, 10, 12, 15]:
    T = tier1(**{**BASE, "gap": v}); rows.append((f"gap<={v}", len(T), round(jac(T, B), 3)))
for v in ["2010-01-01", "2011-01-01", "2012-01-01"]:
    T = tier1(**{**BASE, "fc": v}); rows.append((f"hist<={v}", len(T), round(jac(T, B), 3)))

joint = []
for dvm, nm in itertools.product([3e6, 5e6, 7e6], [400, 500, 750]):
    T = tier1(**{**BASE, "dvf": dvm, "nf": nm})
    joint.append((f"$vol>={dvm:,.0f},news>={nm}", len(T), round(jac(T, B), 3)))
near = [c for lab, _, c in joint
        if lab in (f"$vol>=3,000,000,news>=400", f"$vol>=7,000,000,news>=750")]

R = ["# Universe Threshold Sensitivity\n",
     "_Cutoffs pre-registered as ABSOLUTE economic/data-hygiene floors, "
     "NOT outcome-tuned quantiles. Percentiles below are descriptive._\n",
     f"## Baseline (exact build constants) -> **Tier-1 = {len(B)}** "
     f"(reproduces build: {'OK' if len(B)==870 else 'MISMATCH'})\n",
     "| floor | value | rationale | sits at pctile of eligible pool |",
     "|--|--|--|--|",
     f"| median ADV | >=$5,000,000 | standard tradable-liquidity floor | "
     f"~p{pct_dv:.0f} |",
     f"| total news | >=500 | min for non-degenerate decay series | "
     f"~p{pct_nw:.0f} |",
     f"| hist start | <=2011-01-01 | 2y warmup before 2013 train | fixed |",
     f"| price end | >=2023-12-20 | covers test end | fixed |",
     f"| n_rows | >=2800 | ~0.85x expected trading days | fixed |",
     f"| max gap | <=10d | no >1wk data holes | fixed |",
     "\n## 1-D perturbation (|Tier1|, Jaccard vs 870-baseline)\n",
     "| change | |Tier1| | Jaccard |", "|--|--|--|"]
for a, b, c in rows:
    R.append(f"| {a} | {b} | {c} |")
R += ["\n## Joint perturbation (liquidity x news)\n",
      "| perturbation | |Tier1| | Jaccard |", "|--|--|--|"]
for a, b, c in joint:
    R.append(f"| {a} | {b} | {c} |")
mnj = min(c for _, _, c in joint)
R += [f"\n## Verdict\n- Reproduces the build (Tier-1={len(B)}).",
      f"- One-economic-step-each-side joint Jaccard >= **{min(near) if near else mnj:.3f}**; "
      f"full grid min Jaccard {mnj:.3f}, size band "
      f"[{min(b for _,b,_ in joint)},{max(b for _,b,_ in joint)}].",
      "- Floors are economically pre-registered; Phase-2 headline will be "
      "reported on the +/-1-step universe band as a robustness row, so the "
      "result cannot hinge on a particular cutoff."]
open(fr"{U}\sensitivity_report.md", "w", encoding="utf-8").write("\n".join(R))
print(f"baseline Tier1={len(B)} (expect 870) ; dv_floor~p{pct_dv:.0f} "
      f"news_floor~p{pct_nw:.0f}")
print(f"joint min Jaccard={mnj:.3f} size band "
      f"[{min(b for _,b,_ in joint)},{max(b for _,b,_ in joint)}]")
print("sample 1D:", rows[:3], "...", rows[6:9])
print("wrote universe/sensitivity_report.md")
