from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

BASELINE_EC_PATH = Path("results/baseline_v2_us_equity_curve.csv")
RESULTS_DIR = Path("results")
OUT_PATH = RESULTS_DIR / "spy_benchmark_equity_curve.csv"

def main():
    if not BASELINE_EC_PATH.exists():
        raise FileNotFoundError("Run baseline_v2 first to generate rebalance dates.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ec = pd.read_csv(BASELINE_EC_PATH)
    ec["date"] = pd.to_datetime(ec["date"])
    rebal_dates = ec["date"].sort_values().unique()

    # Download SPY with buffer around dates
    start = (pd.to_datetime(rebal_dates.min()) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    end = (pd.to_datetime(rebal_dates.max()) + pd.Timedelta(days=60)).strftime("%Y-%m-%d")

    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    if spy.empty:
        raise RuntimeError("Failed to download SPY data from yfinance.")

    spy = spy.reset_index().rename(columns={"Date": "date"})
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy[["date", "Close"]].rename(columns={"Close": "spy_close"}).sort_values("date")

    # Map each rebalance date to nearest available trading day (<= date)
    spy_idx = spy.set_index("date")["spy_close"]

    rows = []
    equity = 1.0
    prev_date = None

    for dt in rebal_dates:
        dt = pd.to_datetime(dt)

        # Use last available close on or before dt
        try:
            px_t = float(spy_idx.loc[:dt].iloc[-1])
        except Exception:
            continue

        if prev_date is None:
            prev_date = dt
            prev_px = px_t
            rows.append({"date": dt, "ret": 0.0, "equity": equity, "spy_close": float(px_t)})
            continue

        # Next rebalance date determines holding period return
        # We compute return from prev_date close to dt close (monthly-ish)
        ret = px_t / prev_px - 1.0
        equity *= (1.0 + ret)

        rows.append({"date": dt, "ret": ret, "equity": equity, "spy_close": float(px_t)})

        prev_date = dt
        prev_px = px_t

    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out.to_csv(OUT_PATH, index=False)

    # Quick metrics (monthly)
    monthly = out["ret"].dropna()
    ann_ret = (1.0 + monthly).prod() ** (12.0 / max(1, len(monthly))) - 1.0
    ann_vol = monthly.std() * np.sqrt(12.0)
    sharpe = (monthly.mean() / monthly.std()) * np.sqrt(12.0) if monthly.std() != 0 else np.nan
    mdd = (out["equity"] / out["equity"].cummax() - 1.0).min()

    print("SPY BENCHMARK (aligned to rebalance dates)")
    print(f"Periods: {len(out)}")
    print(f"Final equity: {out['equity'].iloc[-1]:.4f}")
    print(f"Annualized return (approx): {ann_ret:.2%}")
    print(f"Annualized vol (approx): {ann_vol:.2%}")
    print(f"Sharpe (monthly, approx): {sharpe:.2f}")
    print(f"Max drawdown: {mdd:.2%}")
    print(f"Saved: {OUT_PATH}")

if __name__ == "__main__":
    main()
