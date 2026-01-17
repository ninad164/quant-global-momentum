from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw/us_ohlcv.parquet")
OUT_DIR = Path("data/processed")
OUT_PATH = OUT_DIR / "us_features.parquet"

M = 21          # ~1 trading month
M6 = 126        # ~6 months
M12 = 252       # ~12 months

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    g = df.groupby("ticker", group_keys=False)

    # Daily return
    df["ret_1d"] = g["adj_close"].pct_change()

    # Liquidity proxy
    df["dollar_vol"] = df["close"].astype(float) * df["volume"].astype(float)
    df["dollar_vol_20"] = g["dollar_vol"].rolling(20).mean().reset_index(level=0, drop=True)

    # Realized volatility
    df["vol_20"] = g["ret_1d"].rolling(20).std().reset_index(level=0, drop=True)

    # Momentum excluding last month
    df["mom_12_1"] = g["adj_close"].apply(lambda s: s.shift(M) / s.shift(M12) - 1.0)
    df["mom_6_1"]  = g["adj_close"].apply(lambda s: s.shift(M) / s.shift(M6)  - 1.0)

    # Forward return label
    df["fwd_ret_21d"] = g["adj_close"].shift(-M) / df["adj_close"] - 1.0

    out = df[
        [
            "date",
            "ticker",
            "mom_12_1",
            "mom_6_1",
            "vol_20",
            "dollar_vol_20",
            "fwd_ret_21d",
        ]
    ].dropna()

    return out.sort_values(["date", "ticker"]).reset_index(drop=True)

def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError("Missing raw data. Run fetch_data.py first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(RAW_PATH)

    required = {"date", "ticker", "adj_close", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Raw data missing columns: {missing}")

    feats = build_features(df)
    feats.to_parquet(OUT_PATH, index=False)

    print(f"Saved: {OUT_PATH}")
    print(f"Rows: {len(feats):,}")
    print(f"Dates: {feats['date'].nunique():,}")
    print(f"Tickers: {feats['ticker'].nunique():,}")
    print(feats.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
