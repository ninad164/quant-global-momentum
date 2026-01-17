from pathlib import Path
import numpy as np
import pandas as pd

FEATURES_PATH = Path("data/processed/us_features.parquet")
RESULTS_DIR = Path("results")
EQUITY_PATH = RESULTS_DIR / "baseline_us_equity_curve.csv"
METRICS_PATH = RESULTS_DIR / "baseline_us_metrics.txt"

M = 21  # rebalance every ~1 trading month

# Portfolio settings
LONG_Q = 0.90   # top 10%
SHORT_Q = 0.10  # bottom 10%
VOL_EXCLUDE_Q = 0.80   # exclude top 20% vol
LIQ_EXCLUDE_Q = 0.20   # exclude bottom 20% dollar volume

# Costs
COST_BPS = 10  # 10 bps per 1.0 turnover (US-style simple assumption)

def sharpe(daily_ret: pd.Series, ann_factor=252) -> float:
    daily_ret = daily_ret.dropna()
    if daily_ret.std() == 0 or len(daily_ret) < 10:
        return np.nan
    return (daily_ret.mean() / daily_ret.std()) * np.sqrt(ann_factor)

def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return dd.min()

def pick_rebalance_dates(dates: pd.Series, step=M) -> pd.DatetimeIndex:
    unique = pd.Index(pd.to_datetime(dates).unique()).sort_values()
    idx = np.arange(0, len(unique), step)
    return pd.DatetimeIndex(unique[idx])

def compute_weights_for_date(df_date: pd.DataFrame) -> pd.Series:
    """
    Returns weights indexed by ticker for a single rebalance date.
    Long/short equal weight after filters.
    """
    x = df_date.copy()

    # Filters
    vol_cut = x["vol_20"].quantile(VOL_EXCLUDE_Q)
    liq_cut = x["dollar_vol_20"].quantile(LIQ_EXCLUDE_Q)
    x = x[(x["vol_20"] <= vol_cut) & (x["dollar_vol_20"] >= liq_cut)]

    if len(x) < 10:
        return pd.Series(dtype=float)

    # Rank by momentum
    mom = x.set_index("ticker")["mom_12_1"].dropna()

    if len(mom) < 10:
        return pd.Series(dtype=float)

    long_names = mom[mom >= mom.quantile(LONG_Q)].index
    short_names = mom[mom <= mom.quantile(SHORT_Q)].index

    if len(long_names) == 0 or len(short_names) == 0:
        return pd.Series(dtype=float)

    w = pd.Series(0.0, index=mom.index)
    w.loc[long_names] = 1.0 / len(long_names)
    w.loc[short_names] = -1.0 / len(short_names)
    return w

def turnover(prev_w: pd.Series, new_w: pd.Series) -> float:
    """
    0.5 * sum(|delta w|) is a standard turnover proxy.
    """
    all_idx = prev_w.index.union(new_w.index)
    pw = prev_w.reindex(all_idx).fillna(0.0)
    nw = new_w.reindex(all_idx).fillna(0.0)
    return 0.5 * (nw - pw).abs().sum()

def main():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError("Missing features file. Run build_features.py first.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Rebalance dates
    rebal_dates = pick_rebalance_dates(df["date"], step=M)

    # We will realize returns using the label fwd_ret_21d (next-month return)
    # This keeps the backtest simple and avoids needing daily positions.
    equity = 1.0
    equity_curve = []
    prev_w = pd.Series(dtype=float)

    for dt in rebal_dates:
        slice_dt = df[df["date"] == dt]
        if slice_dt.empty:
            continue

        new_w = compute_weights_for_date(slice_dt)
        if new_w.empty:
            continue

        # Portfolio forward return for this holding period
        # Align returns to weights by ticker
        fwd = slice_dt.set_index("ticker")["fwd_ret_21d"]
        port_ret = (new_w.reindex(fwd.index).fillna(0.0) * fwd).sum()

        # Transaction costs via turnover
        t = turnover(prev_w, new_w)
        cost = (COST_BPS / 10000.0) * t  # bps to fraction

        net_ret = port_ret - cost
        equity *= (1.0 + net_ret)

        equity_curve.append(
            {
                "date": dt,
                "gross_ret": float(port_ret),
                "turnover": float(t),
                "cost": float(cost),
                "net_ret": float(net_ret),
                "equity": float(equity),
            }
        )

        prev_w = new_w

    ec = pd.DataFrame(equity_curve)
    if ec.empty:
        raise RuntimeError("Equity curve is empty. Check universe size or filters.")

    ec.to_csv(EQUITY_PATH, index=False)

    # Metrics (convert monthly-ish returns to daily-ish equivalent is messy; report monthly metrics cleanly)
    monthly = ec["net_ret"]
    ann_ret = (1.0 + monthly).prod() ** (12.0 / len(monthly)) - 1.0
    ann_vol = monthly.std() * np.sqrt(12.0)
    sharpe_m = (monthly.mean() / monthly.std()) * np.sqrt(12.0) if monthly.std() != 0 else np.nan
    mdd = max_drawdown(ec["equity"])

    metrics_text = "\n".join(
        [
            "BASELINE US (momentum + vol/liquidity filters) نتائج",
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
        ]
    )

    METRICS_PATH.write_text(metrics_text, encoding="utf-8")
    print(metrics_text)

if __name__ == "__main__":
    main()
