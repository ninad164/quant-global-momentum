# Machine Learning–Driven Quantitative Equity Strategy

This project implements a **systematic, long-only U.S. equity investment strategy** that combines traditional momentum factors with supervised machine learning.
The goal is to evaluate whether ML-based return forecasts can improve **risk-adjusted performance** over a strong baseline and the S&P 500 (SPY).

---

## Project Overview

* **Universe**: Liquid U.S. equities
* **Frequency**: Monthly rebalancing
* **Benchmark**: SPY (S&P 500 ETF)
* **Evaluation**: Rolling out-of-sample backtesting with transaction costs

The project progresses through:

1. A rules-based momentum baseline
2. An improved baseline with risk-aware portfolio construction (Baseline V2)
3. A machine learning strategy using Ridge (L2) regression

---

## Strategies Implemented

### 1. Baseline Strategy

* Cross-sectional momentum ranking
* Volatility and liquidity filters
* Equal or naive weighting

Purpose: Establish a strong, interpretable benchmark.

---

### 2. Baseline V2 (Improved Rules-Based Strategy)

Enhancements over the baseline:

* Inverse-volatility position sizing
* Turnover control and transaction cost modeling
* Optional market regime filter (risk-on / risk-off)

**Result**: Significantly improved drawdowns and Sharpe ratio compared to the naive baseline.

---

### 3. ML Ridge Strategy

A supervised ML approach layered on top of the factor framework.

* **Model**: Ridge Regression (L2-regularized linear model)
* **Features**: Momentum, volatility, liquidity-related signals
* **Target**: Forward 21-day returns
* **Training**: Rolling 60-month window, strictly out-of-sample
* **Portfolio construction**:

  * Select top-ranked predictions
  * Inverse-volatility weighting
  * Liquidity and volatility constraints
  * Weight smoothing to control turnover

---

## Results Summary

| Strategy      | Ann. Return | Ann. Vol | Sharpe | Max Drawdown |
| ------------- | ----------- | -------- | ------ | ------------ |
| Baseline V2   | ~13–15%     | ~13–16%  | ~1.0   | ~22%         |
| ML Ridge      | ~19–20%     | ~17%     | ~1.15  | ~18%         |
| SPY Benchmark | ~13–14%     | ~13%     | ~1.05  | ~21%         |

> All results include transaction cost assumptions and are evaluated out-of-sample.

---

## Key Learnings

* Strong **rules-based baselines** are difficult to beat and should always be established first.
* Machine learning adds value primarily through **better cross-sectional ranking**, not market timing.
* Controlling **overfitting, turnover, and leakage** is more important than model complexity.
* Risk-adjusted metrics (Sharpe, drawdown) matter more than raw returns.

---

## Repository Structure

```
quant-global-momentum/
├── src/
│   ├── fetch_data.py
│   ├── backtest_baseline.py
│   ├── backtest_baseline_v2.py
│   ├── ml_ridge_strategy.py
│   ├── benchmark_spy.py
│   └── reporting scripts
├── universes/
│   └── us_universe.csv
├── results/
│   ├── equity curves
│   ├── performance metrics
│   └── plots
├── README.md
└── CHANGELOG.md
```

---

## Tech Stack

* Python
* Pandas, NumPy
* scikit-learn
* yfinance
* Matplotlib

---

## Disclaimer

This project is for **educational and research purposes only**.
It does not constitute investment advice or a live trading system.

---