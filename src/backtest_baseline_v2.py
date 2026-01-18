from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

FEATURES_PATH = Path("data/processed/us_features.parquet")
RESULTS_DIR = Path("results")
EQUITY_PATH = RESULTS_DIR / "baseline_v2_us_equity_curve.csv"
METRICS_PATH = RESULTS_DIR / "baseline_v2_us_metrics.txt"

M = 21  # ~1 trading month

# Baseline v2 settings (long-only, more realistic and stable)
LONG_Q = 0.80          # top 20%
VOL_EXCLUDE_Q = 0.80   # exclude top 20% vol
LIQ_EXCLUDE_Q = 0.20   # exclude bottom 20% dollar volume

# Costs
COST_BPS = 10  # simple assumption

def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return dd.min()

def pick_rebalance_dates(dates: pd.Series, step=M) -> pd.DatetimeIndex:
    unique = pd.Index(pd.to_datetime(dates).unique()).sort_values()
    idx = np.arange(0, len(unique), step)
    return pd.DatetimeIndex(unique[idx])

def compute_weights_for_date(df_date: pd.DataFrame) -> pd.Series:
    x = df_date.copy()

    # Filters
    vol_cut = x["vol_20"].quantile(VOL_EXCLUDE_Q)
    liq_cut = x["dollar_vol_20"].quantile(LIQ_EXCLUDE_Q)
    x = x[(x["vol_20"] <= vol_cut) & (x["dollar_vol_20"] >= liq_cut)].copy()

    if len(x) < 20:
        return pd.Series(dtype=float)

    # Rank by momentum
    x = x.dropna(subset=["mom_12_1", "vol_20"])
    if len(x) < 20:
        return pd.Series(dtype=float)

    mom_cut = x["mom_12_1"].quantile(LONG_Q)
    x = x[x["mom_12_1"] >= mom_cut].copy()
    if len(x) == 0:
        return pd.Series(dtype=float)

    # Inverse-vol weights (risk-aware)
    inv_vol = 1.0 / x["vol_20"].replace(0, np.nan)
    inv_vol = inv_vol.dropna()
    x = x.loc[inv_vol.index]

    w = inv_vol / inv_vol.sum()
    w.index = x["ticker"].values
    # --- NEW: cap max weight to reduce concentration ---
    MAX_W = 0.10  # 10% cap per stock
    w = w.clip(upper=MAX_W)

    # re-normalize after capping
    if w.sum() != 0:
        w = w / w.sum()
    return w

def turnover(prev_w: pd.Series, new_w: pd.Series) -> float:
    all_idx = prev_w.index.union(new_w.index)
    pw = prev_w.reindex(all_idx).fillna(0.0)
    nw = new_w.reindex(all_idx).fillna(0.0)
    return 0.5 * (nw - pw).abs().sum()

def get_spy_regime(start_date, end_date):
    spy = yf.download("SPY", start=start_date, end=end_date, auto_adjust=True, progress=False)
    if spy.empty:
        raise RuntimeError("Failed to download SPY data for regime detection.")
    spy = spy.reset_index().rename(columns={"Date": "date"})
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.sort_values("date")
    spy["sma_200"] = spy["Close"].rolling(200).mean()
    return spy.set_index("date")[["Close", "sma_200"]]

def main():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError("Missing features file. Run build_features.py first.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    rebal_dates = pick_rebalance_dates(df["date"], step=M)

    # SPY regime filter data (buffer around backtest window)
    start = (rebal_dates.min() - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    end = (rebal_dates.max() + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    spy_regime = get_spy_regime(start, end)

    equity = 1.0
    equity_curve = []
    prev_w = pd.Series(dtype=float)

    for dt in rebal_dates:
        # Regime check: invest only when SPY > 200d SMA
        try:
            spy_row = spy_regime.loc[:dt].iloc[-1]
            # risk_on = float(spy_row["Close"]) > float(spy_row["sma_200"])
            risk_on = float(spy_row["Close"].iloc[0]) > float(spy_row["sma_200"].iloc[0])
        except Exception:
            risk_on = True  # if missing data, default to risk-on

        if not risk_on:
            # Go to cash for this month (zero exposure)
            w = pd.Series(dtype=float)
            t = turnover(prev_w, w)
            cost = (COST_BPS / 10000.0) * t
            net_ret = 0.0 - cost

            equity *= (1.0 + net_ret)
            equity_curve.append(
                {"date": dt, "gross_ret": 0.0, "turnover": float(t),
                "cost": float(cost), "net_ret": float(net_ret), "equity": float(equity)}
            )
            prev_w = w
            continue

        slice_dt = df[df["date"] == dt]
        if slice_dt.empty:
            continue

        new_w = compute_weights_for_date(slice_dt)
        if new_w.empty:
            continue

        # NEW: smooth weights to reduce whipsaw
        if not prev_w.empty:
            all_idx = prev_w.index.union(new_w.index)
            prev_aligned = prev_w.reindex(all_idx).fillna(0.0)
            new_aligned = new_w.reindex(all_idx).fillna(0.0)

            SMOOTH = 0.2  # 20% new weights, 80% old
            w = (1.0 - SMOOTH) * prev_aligned + SMOOTH * new_aligned

            if w.sum() != 0:
                w = w / w.sum()

            w = w[w.abs() > 1e-6]
        else:
            w = new_w

        fwd = slice_dt.set_index("ticker")["fwd_ret_21d"]
        port_ret = (w.reindex(fwd.index).fillna(0.0) * fwd).sum()

        t = turnover(prev_w, w)
        cost = (COST_BPS / 10000.0) * t
        net_ret = port_ret - cost
    
        equity *= (1.0 + net_ret)

        equity_curve.append(
            {"date": dt, "gross_ret": float(port_ret), "turnover": float(t),
             "cost": float(cost), "net_ret": float(net_ret), "equity": float(equity)}
        )
        prev_w = w
    ec = pd.DataFrame(equity_curve)
    if ec.empty:
        raise RuntimeError("Equity curve is empty. Check filters/universe size.")

    ec.to_csv(EQUITY_PATH, index=False)

    monthly = ec["net_ret"]
    ann_ret = (1.0 + monthly).prod() ** (12.0 / len(monthly)) - 1.0
    ann_vol = monthly.std() * np.sqrt(12.0)
    sharpe_m = (monthly.mean() / monthly.std()) * np.sqrt(12.0) if monthly.std() != 0 else np.nan
    mdd = max_drawdown(ec["equity"])

    txt = "\n".join([
        "BASELINE V2 US (long-only momentum + vol/liquidity filters + inverse-vol weights)",
        f"Rebalance periods: {len(ec)}",
        f"Final equity: {ec['equity'].iloc[-1]:.4f}",
        f"Annualized return (approx): {ann_ret:.2%}",
        f"Annualized vol (approx): {ann_vol:.2%}",
        f"Sharpe (monthly, approx): {sharpe_m:.2f}",
        f"Max drawdown: {mdd:.2%}",
        f"Avg turnover: {ec['turnover'].mean():.3f}",
        f"Avg cost per rebalance: {ec['cost'].mean():.4%}",
        "",
        f"Saved equity curve: {EQUITY_PATH}",
    ])
    METRICS_PATH.write_text(txt, encoding="utf-8")
    print(txt)

if __name__ == "__main__":
    main()
