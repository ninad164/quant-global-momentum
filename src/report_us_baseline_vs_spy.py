from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASELINE_PATH = Path("results/baseline_v2_us_equity_curve.csv")
SPY_PATH = Path("results/spy_benchmark_equity_curve.csv")
OUT_FIG1 = Path("results/equity_curve_baseline_v2_vs_spy.png")
OUT_FIG2 = Path("results/drawdown_baseline_v2_vs_spy.png")
OUT_TABLE = Path("results/metrics_baseline_v2_vs_spy.csv")

def max_drawdown(equity: pd.Series) -> float:
    dd = equity / equity.cummax() - 1.0
    return float(dd.min())

def metrics_from_returns(monthly_ret: pd.Series) -> dict:
    monthly_ret = monthly_ret.dropna()
    ann_ret = (1.0 + monthly_ret).prod() ** (12.0 / len(monthly_ret)) - 1.0
    ann_vol = monthly_ret.std() * np.sqrt(12.0)
    sharpe = (monthly_ret.mean() / monthly_ret.std()) * np.sqrt(12.0) if monthly_ret.std() != 0 else np.nan
    return {"ann_return": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe}

def main():
    if not BASELINE_PATH.exists() or not SPY_PATH.exists():
        raise FileNotFoundError("Run baseline_v2 and benchmark_spy first.")

    b = pd.read_csv(BASELINE_PATH)
    s = pd.read_csv(SPY_PATH)

    b["date"] = pd.to_datetime(b["date"])
    s["date"] = pd.to_datetime(s["date"])

    df = b.merge(s[["date", "ret", "equity"]], on="date", how="inner", suffixes=("_strategy", "_spy"))
    df = df.sort_values("date").reset_index(drop=True)

    # Plot equity curves
    plt.figure()
    plt.plot(df["date"], df["equity_strategy"], label="Baseline V2 (Strategy)")
    plt.plot(df["date"], df["equity_spy"], label="SPY Benchmark")
    plt.title("Equity Curve: Baseline V2 vs SPY (Monthly Rebalance Dates)")
    plt.xlabel("Date")
    plt.ylabel("Equity (Start=1.0)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_FIG1, dpi=150)
    plt.close()

    # Drawdowns
    dd_strat = df["equity_strategy"] / df["equity_strategy"].cummax() - 1.0
    dd_spy = df["equity_spy"] / df["equity_spy"].cummax() - 1.0

    plt.figure()
    plt.plot(df["date"], dd_strat, label="Baseline V2 (Strategy)")
    plt.plot(df["date"], dd_spy, label="SPY Benchmark")
    plt.title("Drawdown: Baseline V2 vs SPY")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_FIG2, dpi=150)
    plt.close()

    # Metrics table
    m_strat = metrics_from_returns(df["net_ret"])
    m_spy = metrics_from_returns(df["ret"])
    table = pd.DataFrame(
        [
            {"name": "Baseline V2", **m_strat, "max_drawdown": max_drawdown(df["equity_strategy"])},
            {"name": "SPY", **m_spy, "max_drawdown": max_drawdown(df["equity_spy"])},
        ]
    )
    table.to_csv(OUT_TABLE, index=False)

    print("Saved:")
    print(f"- {OUT_FIG1}")
    print(f"- {OUT_FIG2}")
    print(f"- {OUT_TABLE}")
    print("\nMetrics preview:")
    print(table.to_string(index=False))

if __name__ == "__main__":
    main()
