"""Extract the Tier-1 article corpus for FinBERT scoring.

Streams both news CSVs, keeps rows whose Stock_symbol is in tier1.txt,
dedups the ~2020-06 two-file overlap, and writes a compact parquet:
  universe/tier1_news_corpus.parquet  cols=[symbol, ts, date, title, has_body, src]

Title is the scoring unit (always present; finance-tuned; FNSPID-comparable).
Full timestamp `ts` is kept for point-in-time (T+1) alignment downstream.
"""
import time
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

U = r"D:\Study\FinSharpe\universe"
OUT = fr"{U}\tier1_news_corpus.parquet"
SOURCES = [
    (r"D:\Study\FinSharpe\DATA\All_external.csv", "allext"),
    (r"D:\Study\FinSharpe\DATA\Stock_news\nasdaq_exteral_data.csv", "nasdaq"),
]
USE = ["Date", "Stock_symbol", "Article_title", "Article", "Url"]
CHUNK = 200_000

tier1 = set(pd.read_csv(fr"{U}\tier1.txt", header=None)[0].astype(str)
            .str.strip().str.upper())
print(f"[extract] Tier-1 symbols: {len(tier1)}", flush=True)

schema = pa.schema([("symbol", pa.string()), ("ts", pa.string()),
                    ("date", pa.string()), ("title", pa.string()),
                    ("has_body", pa.bool_()), ("src", pa.string())])
writer = pq.ParquetWriter(OUT, schema, compression="zstd")

seen = set()           # strict composite key: sym|date|normtitle|normurl
kept = 0
t0 = time.time()
for path, tag in SOURCES:
    n = 0
    rdr = pd.read_csv(path, usecols=USE, dtype=str, chunksize=CHUNK,
                      engine="c", on_bad_lines="skip", encoding_errors="replace")
    for ci, ch in enumerate(rdr):
        ch = ch.dropna(subset=["Stock_symbol", "Date"])
        ch["sym"] = ch["Stock_symbol"].astype(str).str.strip().str.upper()
        ch = ch[ch["sym"].isin(tier1)]
        if len(ch):
            ch["ts"] = ch["Date"].astype(str).str.strip()
            ch["date"] = ch["ts"].str.slice(0, 10)
            ch = ch[ch["date"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").fillna(False)]
            title = ch["Article_title"].fillna("").astype(str).str.strip()
            body = ch["Article"].fillna("").astype(str).str.strip()
            # title fallback: first ~60 words of body when title is missing
            need = title.str.len() == 0
            title = title.mask(need, body.str.split().str[:60].str.join(" "))
            ch = ch.assign(title=title, has_body=body.str.len() > 20)
            ch = ch[ch["title"].str.len() > 0]
            # STRICT dedup key (Jury5#6 fix): symbol | date | FULL
            # normalized title | normalized url. The old title[:120] key
            # over-merged 5.82% of distinct same-day stories.
            ntitle = (ch["title"].astype(str).str.lower()
                      .str.replace(r"\s+", " ", regex=True).str.strip())
            nurl = (ch["Url"].fillna("").astype(str).str.lower().str.strip()
                    .str.split("?").str[0].str.split("#").str[0]
                    .str.rstrip("/"))
            keys = [hash((s, d, t, u)) for s, d, t, u in
                    zip(ch["sym"], ch["date"], ntitle, nurl)]
            rec = []
            for sy, ts_, dt, ti, hb, k in zip(
                    ch["sym"], ch["ts"], ch["date"], ch["title"],
                    ch["has_body"], keys):
                if k in seen:
                    continue
                seen.add(k)
                rec.append((sy, ts_, dt, ti, bool(hb), tag))
            if rec:
                tb = pa.Table.from_pylist(
                    [dict(symbol=a, ts=b, date=c, title=d, has_body=e, src=f)
                     for a, b, c, d, e, f in rec], schema=schema)
                writer.write_table(tb)
                kept += len(rec)
        n += len(ch)
        if (ci + 1) % 25 == 0:
            print(f"[extract] {tag} ~{n:,} matched-rows kept={kept:,} "
                  f"{time.time()-t0:,.0f}s", flush=True)
    print(f"[extract] DONE {tag} kept_total={kept:,} {time.time()-t0:,.0f}s",
          flush=True)

writer.close()
df = pd.read_parquet(OUT, columns=["symbol", "date"])
print(f"[extract] wrote {OUT}: {len(df):,} articles, "
      f"{df['symbol'].nunique()} symbols, "
      f"{df['date'].min()}..{df['date'].max()} elapsed {time.time()-t0:,.0f}s",
      flush=True)
