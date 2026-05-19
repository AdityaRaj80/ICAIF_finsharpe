"""Build the tiered stock universe from price+news profiles.

Modeling window (see PREREGISTRATION.md §2-3 — authoritative):
  feature warmup    : 2011-01-01 .. 2012-12-31  (>=504-day lookback + alpha)
  inner-CPCV pool   : 2013-01-01 .. 2020-12-31  (model select/HPO = nested
                       inner folds only; NO separate calendar-2020 val block)
  held-out test     : 2021-01-01 .. 2023-12-28
  (purge/embargo = H td, per-horizon, handled at the CV stage)

Selection criteria are ABSOLUTE economic / data-hygiene FLOORS, pre-set
a-priori and frozen (PREREGISTRATION.md §2) — NOT outcome-tuned quantiles.
The empirical percentile each floor happens to sit at is reported
descriptively in universe/sensitivity_report.md (~p47 ADV, ~p42 news);
that report also shows membership is stable to +/-1-step perturbation.
  price first_date  <= 2011-01-01            (full warmup coverage)
  price last_date   >= 2023-12-20            (covers test end)
  n_rows            >= 2800                  (data-completeness hygiene)
  max_gap_days      <= 10                    (no >1-week data holes)
  med_dollar_vol_recent252 >= 5_000_000      (standard tradable-liquidity floor)
  total news        >= 500                   (min for non-degenerate decay)
  news in >=5 of 7 'train-era' years (2013-2019)
  news in >=2 of 3 'test-era' years (2021-2023)  (2021 trough disclosed)
  not in curated ETF/fund blocklist or leveraged/inverse regex
"""
import json, re
import numpy as np
import pandas as pd

U = r"D:\Study\FinSharpe\universe"
px = pd.read_csv(fr"{U}\price_profile.csv")
nw = pd.read_csv(fr"{U}\news_profile.csv")

CRIT = dict(
    first_date_max="2011-01-01", last_date_min="2023-12-20",
    n_rows_min=2800, max_gap_days_max=10, dollar_vol_min=5_000_000,
    news_total_min=500, train_years=list(range(2013, 2020)),
    val_years=[2020], test_years=[2021, 2022, 2023],
    train_years_min=5, test_years_min=2,
)

# Curated ETF / ETN / fund / leveraged blocklist (no security master available;
# heuristic, conservative — documented as a known limitation).
ETF = set("""
SPY QQQ DIA IWM IVV VOO VTI VEA VWO IEFA EEM EFA IEMG SCHB SCHX SCHD SDY DVY
VYM VIG NOBL RSP MDY SLY IJH IJR IVW IVE VUG VTV MTUM QUAL USMV SPLV ACWI VT
AGG BND BNDX LQD HYG JNK TLT IEF SHY TIP EMB PFF BKLN VCIT VCSH VGSH MUB GOVT
GLD SLV IAU USO UNG DBC PDBC GDX GDXJ SIL PPLT PALL
XLE XLF XLK XLV XLI XLY XLP XLU XLB XLRE XLC SMH SOXX XBI IBB ARKK ARKG ARKW
ARKQ ARKF VGT VHT VFH VNQ VDE VAW VIS VOX VPU VCR VDC IYR ITB XHB XOP OIH KRE
KBE XME XRT IGV HACK SKYY FDN PNQI ROBO BOTZ FINX
FXI MCHI INDA EWZ EWW EWG EWU EWC EWA EWH EWS EWT EWY EWP EWI EWL EWN EWD EWQ
EWK EWO EWM EZA EPHE THD TUR GREK ARGT EWJ GXC PGJ KWEB ASHR EIDO EPOL EWZS
RSX VGK VPL EEMA EEMS SCZ SCHF GWX EWUS DXJ HEDJ EPI
TQQQ SQQQ UVXY VXX SVXY SPXU SPXL SSO UPRO TNA TZA FAS FAZ SOXL SOXS LABU LABD
TLT TMF TMV YINN YANG NUGT DUST JNUG JDST UCO SCO BOIL KOLD UVIX SVIX
SCHA SCHM SCHG SCHV SCHP SCHO SCHR SCHZ SPTM SPLG SPMD SPSM ESGU SUSA ICLN TAN
PBW FAN QCLN LIT REMX URA NLR PICK COPX WOOD CUT MOO VEGI FTGC GSG BCI COMT
DGRO HDV SPHD SPYD FDL FVD VONG VONV IWF IWD IWN IWO IWP IWS IWR IWV IWB OEF
JEPI JEPQ DIVO QYLD RYLD XYLD NUSI SCHH USRT REZ MORT REM
""".split())
LEV_RE = re.compile(r"^(TQQQ|SQQQ|UVXY|VXX|SVXY|.*(BULL|BEAR)).*$")

px["first_d"] = pd.to_datetime(px["first_date"], errors="coerce")
px["last_d"] = pd.to_datetime(px["last_date"], errors="coerce")
m = px.merge(nw, on="ticker", how="left")
ycols = [f"y{y}" for y in range(2009, 2026)]
for c in ycols + ["total", "allext_count", "nasdaq_count"]:
    m[c] = m[c].fillna(0)

def reason(r):
    if pd.isna(r.first_d) or r.first_d > pd.Timestamp(CRIT["first_date_max"]):
        return "history_starts_too_late"
    if pd.isna(r.last_d) or r.last_d < pd.Timestamp(CRIT["last_date_min"]):
        return "price_ends_before_test_end(2020_freeze_or_delist)"
    if r.n_rows < CRIT["n_rows_min"]:
        return "too_few_price_rows"
    if r.max_gap_days > CRIT["max_gap_days_max"] or r.max_gap_days < 0:
        return "price_gap_too_large"
    if r.med_dollar_vol_recent252 < CRIT["dollar_vol_min"]:
        return "illiquid"
    if r.total < CRIT["news_total_min"]:
        return "too_few_news_articles"
    tr = sum(1 for y in CRIT["train_years"] if r[f"y{y}"] > 0)
    te = sum(1 for y in CRIT["test_years"] if r[f"y{y}"] > 0)
    if tr < CRIT["train_years_min"]:
        return "sparse_news_in_train"
    if te < CRIT["test_years_min"]:
        return "sparse_news_in_test"
    if r.ticker in ETF or LEV_RE.match(str(r.ticker)):
        return "etf_or_fund_or_leveraged"
    return "PASS"

m["exclusion_reason"] = m.apply(reason, axis=1)
m["news_window"] = m[[f"y{y}" for y in range(2011, 2024)]].sum(axis=1).astype(int)
t1 = m[m.exclusion_reason == "PASS"].copy().sort_values(
    "med_dollar_vol_recent252", ascending=False)

# Tier 2: most sentiment-dense subset of Tier 1 (top by in-window articles)
t2 = t1.sort_values("news_window", ascending=False).head(150)

t1["ticker"].to_csv(fr"{U}\tier1.txt", index=False, header=False)
t2["ticker"].to_csv(fr"{U}\tier2.txt", index=False, header=False)
cols = ["ticker", "first_date", "last_date", "n_rows", "max_gap_days",
        "med_dollar_vol_recent252", "total", "allext_count", "nasdaq_count",
        "news_window", "exclusion_reason"]
m[cols].sort_values("exclusion_reason").to_csv(
    fr"{U}\universe_candidates.csv", index=False)

spec = dict(
    created="2026-05-19",
    dataset="FNSPID (contaminated-replication arm; PREREG amendment c)",
    scope="C1-only; sentiment de-scoped to optional feature",
    window=dict(warmup="2011-01-01..2012-12-31",
                inner_cpcv_pool="2013-01-01..2020-12-31",
                norm_fit_subwindow="2013-01-01..2019-12-31",
                test="2021-01-01..2023-12-28",
                note="NO separate calendar-2020 validation block; model "
                     "selection/HPO via nested CPCV inner folds only "
                     "(PREREGISTRATION.md §3)"),
    criteria=CRIT,
    counts=dict(
        all_price_tickers=int(len(px)),
        symbols_with_news=int(len(nw)),
        tier1=int(len(t1)), tier2=int(len(t2)),
        excluded={k: int(v) for k, v in
                  m.exclusion_reason.value_counts().items() if k != "PASS"},
    ),
    known_limitations=[
        "Survivorship: Tier-1 requires price through 2023; ~3,232 names end "
        "at the FNSPID mid-2020 data freeze (artifact, not real delisting). "
        "Only 2 names stop genuinely mid-history -> NO clean delisted control "
        "is internally constructible. Absolute Sharpe is optimistic; defensible "
        "claims are RELATIVE (model vs model, overlay on/off).",
        "2021 news trough (~182k vs ~1.1M in 2018-19): test years 2021-22 are "
        "news-sparse; report sentiment results per-year, never pooled-only.",
        "News two-file splice (~2020-06): All_external ends ~2020, nasdaq "
        "carries 2021-2023; dedup by (symbol,date,url) when building features.",
        "ETF/fund screen is a curated heuristic blocklist (no security master); "
        "residual small-cap-lookalike contamination possible.",
    ],
)
with open(fr"{U}\universe_spec.json", "w") as f:
    json.dump(spec, f, indent=2)

print("TIER1:", len(t1), " TIER2:", len(t2))
print("Exclusion breakdown:")
print(m.exclusion_reason.value_counts().to_string())
print("\nTier1 liquidity (med_dollar_vol_recent252) describe:")
print(t1["med_dollar_vol_recent252"].describe().to_string())
print("\nTier1 news 'total' describe:")
print(t1["total"].describe().to_string())
print("\nTier1 head (most liquid):", list(t1["ticker"].head(15)))
print("Wrote tier1.txt, tier2.txt, universe_candidates.csv, universe_spec.json")
