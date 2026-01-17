from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

FEATURES_PATH = Path("data/processed/us_features.parquet")
RESULTS_DIR = Path("results")
EQUITY_PATH = RESULTS_DIR / "ml_ridge_us_equity_curve.csv"
METRICS_PATH = RESULTS_DIR / "ml_ridge_us_metrics.txt"

# Rebalance and training settings
M = 21                  # ~1 trading month
TRAIN_MONTHS = 60       # rolling training window
TOP_Q = 0.80            # long top 20%
VOL_EXCLUDE_Q = 0.80    # exclude top 20% vol
LIQ_EXCLUDE_Q = 0.20    # exclude bottom 20% liquidity
COST_BPS = 10           # transaction cost assumption

FEATURE_COLS = ["mom_12_1", "mom_6_1", "vol_20", "dollar_vol_20"]
LABEL_COL = "fwd_ret_21d"

def pick_rebalance_dates(dates: pd.Series, step=M) -> pd.DatetimeIndex:
    unique = pd.Index(pd.to_datetime(dates).unique()).sort_values()
    idx = np.arange(0, len(unique), step)
    return pd.DatetimeIndex(unique[idx])

def cross_sectional_zscore(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        mu = out[c].mean()
        sd = out[c].std()
        out[c] = (out[c] - mu) / sd if sd != 0 else 0.0
    return out

def turnover(prev_w: pd.Series, new_w: pd.Series) -> float:
    all_idx = prev_w.index.union(new_w.index)
    pw = prev_w.reindex(all_idx).fillna(0.0)
    nw = new_w.reindex(all_idx).fillna(0.0)
    return 0.5 * (nw - pw).abs().sum()

def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min())

def compute_weights(df: pd.DataFrame) -> pd.Series:
    x = df.copy()

    # Risk & liquidity filters
    vol_cut = x["vol_20"].quantile(VOL_EXCLUDE_Q)
    liq_cut = x["dollar_vol_20"].quantile(LIQ_EXCLUDE_Q)
    x = x[(x["vol_20"] <= vol_cut) & (x["dollar_vol_20"] >= liq_cut)]

    if len(x) < 20:
        return pd.Series(dtype=float)

    # Select top predicted stocks
    cut = x["pred"].quantile(TOP_Q)
    x = x[x["pred"] >= cut]

    if x.empty:
        return pd.Series(dtype=float)

    # Inverse volatility weighting
    inv_vol = 1.0 / x["vol_20"].replace(0, np.nan)
    inv_vol = inv_vol.dropna()
    x = x.loc[inv_vol.index]

    w = inv_vol / inv_vol.sum()
    w.index = x["ticker"].values
    return w

def main():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError("Run build_features.py first.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    needed = ["date", "ticker"] + FEATURE_COLS + [LABEL_COL]
    df = df[needed].dropna().reset_index(drop=True)

    rebal_dates = pick_rebalance_dates(df["date"])

    equity = 1.0
    prev_w = pd.Series(dtype=float)
    curve = []

    for i in range(TRAIN_MONTHS, len(rebal_dates)):
        train_start = rebal_dates[i - TRAIN_MONTHS]
        train_end = rebal_dates[i - 1]
        test_date = rebal_dates[i]

        train = df[(df["date"] >= train_start) & (df["date"] <= train_end)].copy()
        test = df[df["date"] == test_date].copy()

        if train.empty or test.empty:
            continue

        # Cross-sectional standardization
        train = train.groupby("date", group_keys=False).apply(
            lambda x: cross_sectional_zscore(x, FEATURE_COLS)
        )
        test = cross_sectional_zscore(test, FEATURE_COLS)

        X_train = train[FEATURE_COLS].values
        y_train = train[LABEL_COL].values
        X_test = test[FEATURE_COLS].values

        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)

        test["pred"] = model.predict(X_test)

        w = compute_weights(test[["ticker", "pred", "vol_20", "dollar_vol_20"]])
        if w.empty:
            continue

        fwd = test.set_index("ticker")[LABEL_COL]
        gross_ret = float((w.reindex(fwd.index).fillna(0.0) * fwd).sum())

        t = turnover(prev_w, w)
        cost = (COST_BPS / 10000.0) * t
        net_ret = gross_ret - cost

        equity *= (1.0 + net_ret)

        curve.append({
            "date": test_date,
            "gross_ret": gross_ret,
            "turnover": float(t),
            "cost": float(cost),
            "net_ret": net_ret,
            "equity": equity
        })

        prev_w = w

    ec = pd.DataFrame(curve)
    if ec.empty:
        raise RuntimeError("No ML results generated.")

    ec.to_csv(EQUITY_PATH, index=False)

    monthly = ec["net_ret"]
    ann_ret = (1.0 + monthly).prod() ** (12.0 / len(monthly)) - 1.0
    ann_vol = monthly.std() * np.sqrt(12.0)
    sharpe = (monthly.mean() / monthly.std()) * np.sqrt(12.0)
    mdd = max_drawdown(ec["equity"])

    txt = "\n".join([
        "ML RIDGE STRATEGY (rolling out-of-sample)",
        f"Train window (months): {TRAIN_MONTHS}",
        f"Rebalance periods: {len(ec)}",
        f"Final equity: {ec['equity'].iloc[-1]:.4f}",
        f"Annualized return (approx): {ann_ret:.2%}",
        f"Annualized vol (approx): {ann_vol:.2%}",
        f"Sharpe (monthly, approx): {sharpe:.2f}",
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
