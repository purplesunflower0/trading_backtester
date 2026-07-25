"""
Phase 6: Visualization
--------------------------------------------------------------
Generates the charts that actually tell the project's story:
  1. Equity curve comparison (strategy vs buy-and-hold) per asset,
     with the 2024+ holdout period shaded so it's visually obvious
     which part of the chart is "out of sample."
  2. Drawdown chart per asset — visualizes the max drawdown numbers
     from Phase 5 instead of just stating them as a percentage.
  3. A summary bar chart comparing Sharpe and Calmar ratios across
     all assets, strategy vs buy-and-hold, side by side.

All charts are saved as PNGs under reports/ so they can be dropped
straight into a resume/portfolio writeup or README.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

HOLDOUT_START = "2024-01-01"
ASSETS = ["gold", "crude", "sp500"]
STARTING_CAPITAL = 100_000.0
TRANSACTION_COST_PCT = 0.001  # must match Phase 4/5's assumption

OUT_DIR = Path("reports")


def buy_and_hold_equity_curve(price_df: pd.DataFrame, start_date) -> pd.Series:
    window = price_df.loc[price_df.index >= start_date]
    if len(window) < 2:
        return pd.Series(dtype=float)
    entry_price = window["Close"].iloc[0]
    entry_cost = STARTING_CAPITAL * TRANSACTION_COST_PCT
    units = (STARTING_CAPITAL - entry_cost) / entry_price
    equity = units * window["Close"]
    equity.name = "equity"
    return equity


def drawdown_series(equity: pd.Series) -> pd.Series:
    running_max = equity.cummax()
    return (equity / running_max - 1) * 100


def plot_equity_comparison(asset: str, strat_equity: pd.Series, bh_equity: pd.Series):
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(strat_equity.index, strat_equity.values, label="Strategy", linewidth=1.6)
    ax.plot(bh_equity.index, bh_equity.values, label="Buy & Hold", linewidth=1.6, alpha=0.85)

    # Shade the holdout period so it's visually obvious what's out-of-sample
    ax.axvspan(pd.Timestamp(HOLDOUT_START), strat_equity.index.max(),
               color="gray", alpha=0.12, label="Holdout (2024-present)")

    ax.set_title(f"{asset.upper()} — Equity Curve: Strategy vs Buy & Hold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{asset}_equity_comparison.png", dpi=150)
    plt.close(fig)


def plot_drawdown_comparison(asset: str, strat_equity: pd.Series, bh_equity: pd.Series):
    strat_dd = drawdown_series(strat_equity)
    bh_dd = drawdown_series(bh_equity)

    fig, ax = plt.subplots(figsize=(11, 4))

    ax.fill_between(strat_dd.index, strat_dd.values, 0, label="Strategy", alpha=0.5)
    ax.fill_between(bh_dd.index, bh_dd.values, 0, label="Buy & Hold", alpha=0.35)

    ax.axvspan(pd.Timestamp(HOLDOUT_START), strat_dd.index.max(),
               color="gray", alpha=0.10)

    ax.set_title(f"{asset.upper()} — Drawdown Over Time (Strategy vs Buy & Hold)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{asset}_drawdown_comparison.png", dpi=150)
    plt.close(fig)


def plot_summary_bars(risk_df: pd.DataFrame):
    holdout_label = f"Holdout only ({HOLDOUT_START}-present)"
    subset = risk_df[risk_df["period"] == holdout_label]

    metrics_to_plot = ["sharpe", "calmar"]
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(12, 4.5))

    for ax, metric in zip(axes, metrics_to_plot):
        pivot = subset.pivot(index="asset", columns="series", values=metric)
        pivot = pivot.reindex(ASSETS)
        pivot.plot(kind="bar", ax=ax, rot=0)
        ax.set_title(f"{metric.capitalize()} Ratio — Holdout Period")
        ax.set_ylabel(metric.capitalize())
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "summary_sharpe_calmar_holdout.png", dpi=150)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    risk_df = pd.read_csv("data_processed/risk_metrics.csv")

    for asset in ASSETS:
        strat_equity = pd.read_csv(
            f"data_processed/{asset}_equity_curve.csv", index_col="date", parse_dates=True
        )["equity"]

        price_df = pd.read_csv(
            f"data_raw/{asset}.csv", index_col="Date", parse_dates=True
        ).sort_index()
        bh_equity = buy_and_hold_equity_curve(price_df, price_df.index.min())

        plot_equity_comparison(asset, strat_equity, bh_equity)
        plot_drawdown_comparison(asset, strat_equity, bh_equity)
        print(f"{asset}: saved equity + drawdown charts")

    plot_summary_bars(risk_df)
    print("Saved summary Sharpe/Calmar bar chart")

    print(f"\nAll charts saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()