"""
Phase 5: Risk Metrics
Computes Sharpe, Sortino, max drawdown, Calmar, and Value-at-Risk (VaR)
for both the strategy and the buy-and-hold benchmark, on the full period
and the 2024-present holdout, per asset.

This is where we find out if the strategy earns its keep despite losing
the raw-return race in Phase 4: even a strategy with lower total return
can be genuinely better if it loses much less money during bad periods.
  - Sharpe ratio: return earned per unit of TOTAL volatility (up and down
    swings both count against you here). Higher = smoother, more
    consistent gains relative to how bumpy the ride was.
  - Sortino ratio: like Sharpe, but only penalizes DOWNSIDE volatility
    (bad swings). Doesn't punish a strategy for being volatile on the way
    UP. Usually higher than Sharpe for the same strategy.
  - Max drawdown: the single worst peak-to-trough loss the strategy ever
    experienced, in percentage terms. E.g. -30% means at some point your
    portfolio value fell 30% below its highest point before recovering.
    This is the number that answers "how bad could it have felt to hold
    this."
  - Calmar ratio: annualized return divided by max drawdown. Answers
    "how much return did I get per unit of worst-case pain." Higher is
    better; a strategy with modest return but a tiny max drawdown can beat
    one with high return but a terrifying max drawdown.
  - VaR (Value at Risk, 95%): on the worst 5% of days historically, you
    lost at least this much (as a % of portfolio value) in a single day.
    A quick, standard way to describe "how bad can one day get."
"""

import pandas as pd
import numpy as np
from pathlib import Path

TRADING_DAYS_PER_YEAR = 252
HOLDOUT_START = "2024-01-01"

ASSETS = ["gold", "crude", "sp500"]

STARTING_CAPITAL = 100_000.0
TRANSACTION_COST_PCT = 0.001 


# Build a daily equity curve for buy-and-hold (needed for drawdown/Sharpe,
# not just the single start/end numbers Phase 4 printed)

def buy_and_hold_equity_curve(price_df: pd.DataFrame, start_date: str) -> pd.Series:
    window = price_df.loc[price_df.index >= start_date]
    if len(window) < 2:
        return pd.Series(dtype=float)

    entry_price = window["Close"].iloc[0]
    entry_cost = STARTING_CAPITAL * TRANSACTION_COST_PCT
    units = (STARTING_CAPITAL - entry_cost) / entry_price

    equity = units * window["Close"]
    equity.name = "equity"
    return equity


# Risk metric calculations

def compute_metrics(equity: pd.Series) -> dict:
    equity = equity.dropna()
    if len(equity) < 2:
        return None

    daily_returns = equity.pct_change().dropna()
    n_days = len(equity)

    # Annualized return, based on how many years the equity curve actually spans
    years = n_days / TRADING_DAYS_PER_YEAR
    total_growth = equity.iloc[-1] / equity.iloc[0]
    annualized_return = total_growth ** (1 / years) - 1 if years > 0 else np.nan

    # Sharpe: reward per unit of total volatility
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
              if daily_returns.std() > 0 else 0.0)

    # Sortino: reward per unit of DOWNSIDE volatility only
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std()
    sortino = (daily_returns.mean() / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR)
               if downside_std and downside_std > 0 else 0.0)

    # Max drawdown: worst peak-to-trough decline
    running_max = equity.cummax()
    drawdown = (equity / running_max) - 1
    max_drawdown = drawdown.min()

    # Calmar: annualized return per unit of worst-case pain
    calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

    # Historical VaR at 95%: the loss threshold for the worst 5% of days
    var_95 = daily_returns.quantile(0.05)

    return {
        "annualized_return": annualized_return * 100,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown * 100,
        "calmar": calmar,
        "var_95_daily": var_95 * 100,
    }


def print_metrics(metrics: dict, label: str):
    if metrics is None:
        print(f"    {label}: not enough data")
        return
    print(f"    {label}:")
    print(f"      Annualized return: {metrics['annualized_return']:.2f}%")
    print(f"      Sharpe:  {metrics['sharpe']:.3f}")
    print(f"      Sortino: {metrics['sortino']:.3f}")
    print(f"      Max drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"      Calmar: {metrics['calmar']:.3f}")
    print(f"      VaR 95% (1-day): {metrics['var_95_daily']:.2f}%")


# Main

def main():
    out_rows = []

    for asset in ASSETS:
        print(f"\n{asset.upper()}")

        # --- Strategy equity curve (from Phase 4's saved output) ---
        strat_path = Path(f"data_processed/{asset}_equity_curve.csv")
        strat_equity = pd.read_csv(strat_path, index_col="date", parse_dates=True)["equity"]

        # --- Buy-and-hold equity curve (built fresh from raw prices) ---
        price_df = pd.read_csv(f"data_raw/{asset}.csv", index_col="Date", parse_dates=True).sort_index()

        for label, start in [("Full period (2016-present)", price_df.index.min()),
                              (f"Holdout only ({HOLDOUT_START}-present)", HOLDOUT_START)]:

            print(f"  {label}")

            strat_window = strat_equity.loc[strat_equity.index >= start]
            strat_metrics = compute_metrics(strat_window)
            print("   STRATEGY:")
            print_metrics(strat_metrics, "")

            bh_equity = buy_and_hold_equity_curve(price_df, start)
            bh_metrics = compute_metrics(bh_equity)
            print("   BUY-AND-HOLD:")
            print_metrics(bh_metrics, "")

            for name, m in [("strategy", strat_metrics), ("buy_and_hold", bh_metrics)]:
                if m is not None:
                    out_rows.append({"asset": asset, "period": label, "series": name, **m})

    results_df = pd.DataFrame(out_rows)
    out_path = Path("data_processed/risk_metrics.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved full risk metrics table to {out_path}")


if __name__ == "__main__":
    main()