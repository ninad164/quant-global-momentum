from pathlib import Path
import pandas as pd
import yfinance as yf

RAW_DIR = Path("data/raw")
UNIVERSE_PATH = Path("universes/us_universe.csv")

def load_universe(path: Path) -> list[str]:
    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        raise ValueError("Universe CSV must have a 'ticker' column.")
    tickers = df["ticker"].dropna().astype(str).str.strip().tolist()
    if not tickers:
        raise ValueError("Universe is empty.")
    return tickers

def download_ohlcv(tickers: list[str], start="2010-01-01") -> pd.DataFrame:
    df = yf.download(
        tickers=tickers,
        start=start,
        auto_adjust=False,
        group_by="ticker",
        progress=True,
        threads=True,
    )
    if df.empty:
        raise RuntimeError("Downloaded data is empty. Check tickers/network.")
    return df

def to_long_panel(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for ticker in df.columns.levels[0]:
        sub = df[ticker].copy()
        sub.columns = [c.lower().replace(" ", "_") for c in sub.columns]
        sub = sub.reset_index().rename(columns={"Date": "date", "date": "date"})
        sub["ticker"] = ticker
        out.append(sub)
    panel = pd.concat(out, ignore_index=True)

    required = {"date","ticker","open","high","low","close","adj_close","volume"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"Missing columns after conversion: {missing}")

    panel["date"] = pd.to_datetime(panel["date"])
    return panel.sort_values(["ticker","date"]).reset_index(drop=True)

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    tickers = load_universe(UNIVERSE_PATH)
    print(f"Loaded {len(tickers)} tickers.")

    wide = download_ohlcv(tickers, start="2010-01-01")
    panel = to_long_panel(wide)

    out_path = RAW_DIR / "us_ohlcv.parquet"
    panel.to_parquet(out_path, index=False)
    print(f"Saved: {out_path} | rows={len(panel):,} | tickers={panel['ticker'].nunique()}")

if __name__ == "__main__":
    main()
