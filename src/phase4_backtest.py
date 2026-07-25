"""
Phase 4: Backtest Engine with Transaction Costs
Simulates an actual portfolio trading the MA/RSI strategy, using each
asset's best (ma_short, ma_long) combo picked automatically from Phase 3's
optimization_results.csv.
  - "Equity curve" = portfolio value over time, day by day. This is the
    core output everything else (Sharpe, drawdown, etc. in Phase 5) is
    built from.
  - "Transaction cost" = money lost every time we buy or sell, simulating
    commission + slippage. Charged as a % of the trade's dollar value.
  - "All-in/all-out" sizing = when the signal says go long, we put 100% of
    current cash into the asset; when it says exit, we sell everything
    back to cash. No partial positions, no leverage.
  - We run signals on the FULL price history (so rolling MA/RSI windows
    have enough lookback), but we separately report performance on just
    the 2024-present holdout window, since that's the truly untouched
    out-of-sample test. The 2015-2023 numbers are shown too, but Phase 3
    already validated that period across folds — 2024+ is the real proof.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from signals import build_signals

ASSETS = {
    "gold": "data_raw/gold.csv",
    "crude": "data_raw/crude.csv",
    "sp500": "data_raw/sp500.csv",
}

OPTIMIZATION_RESULTS_PATH = "data_processed/optimization_results.csv"

STARTING_CAPITAL = 100_000.0
TRANSACTION_COST_PCT = 0.001  # 0.1% per trade (buy or sell), covers commission + slippage

HOLDOUT_START = "2024-01-01"  # untouched final test window

RSI_PERIOD = 14
OVERSOLD = 30
OVERBOUGHT = 70


# Step 1: figure out each asset's best params from Phase 3's results


def get_best_params_per_asset(results_path: str) -> dict:
    """Reads Phase 3's grid search output and picks the best (ma_short, ma_long)
    per asset, ranked by avg_sharpe first, worst_fold_sharpe as tiebreaker —
    same ranking rule Phase 3 used."""
    df = pd.read_csv(results_path)
    best = {}
    for asset, group in df.groupby("asset"):
        top = group.sort_values(
            by=["avg_sharpe", "worst_fold_sharpe"], ascending=False
        ).iloc[0]
        best[asset] = {"ma_short": int(top["ma_short"]), "ma_long": int(top["ma_long"])}
    return best


# Step 2: simulate the portfolio day by day

def run_backtest(df: pd.DataFrame, ma_short: int, ma_long: int) -> pd.DataFrame:
    """
    Walks through the price history day by day. Whenever `signal` says
    enter (+1), we spend all our cash buying the asset (minus transaction
    cost). Whenever `signal` says exit (-1), we sell everything back to
    cash (minus transaction cost). Otherwise we just hold whatever we're
    already holding and let its value move with the price.
    """
    signaled = build_signals(df, ma_short, ma_long, RSI_PERIOD, OVERSOLD, OVERBOUGHT)

    cash = STARTING_CAPITAL
    units = 0.0  # how many "shares"/units of the asset we currently hold
    equity_curve = []
    trade_log = []

    for date, row in signaled.iterrows():
        price = row["Close"]
        signal = row["signal"]

        if signal == 1 and units == 0:
            # Enter: spend all cash on the asset, paying a transaction cost
            cost = cash * TRANSACTION_COST_PCT
            spendable = cash - cost
            units = spendable / price
            trade_log.append({"date": date, "action": "BUY", "price": price,
                               "units": units, "cost": cost})
            cash = 0.0

        elif signal == -1 and units > 0:
            # Exit: sell everything back to cash, paying a transaction cost
            proceeds = units * price
            cost = proceeds * TRANSACTION_COST_PCT
            cash = proceeds - cost
            trade_log.append({"date": date, "action": "SELL", "price": price,
                               "units": units, "cost": cost})
            units = 0.0

        # Mark-to-market: what is the portfolio worth right now?
        portfolio_value = cash + units * price
        equity_curve.append({"date": date, "equity": portfolio_value})

    equity_df = pd.DataFrame(equity_curve).set_index("date")
    trades_df = pd.DataFrame(trade_log)
    return equity_df, trades_df


# Step 2b: buy-and-hold benchmark

def compute_buy_and_hold(df: pd.DataFrame, start_date: str) -> dict:
    """
    The simplest possible strategy: buy at the start of the window and never
    sell. This is the benchmark every 'active' strategy has to beat to
    justify its own complexity and trading costs — if buy-and-hold wins,
    all that signal logic isn't adding value, it's just adding noise (and
    fees) on top of what the market did anyway.

    We pay ONE transaction cost on the way in (buying), and never sell, so
    the ending value is just marked-to-market at the last available price —
    matching exactly how the strategy's own equity curve is measured if it
    happens to still be holding a position at the end.
    """
    window = df.loc[df.index >= start_date]
    if len(window) < 2:
        return None

    entry_price = window["Close"].iloc[0]
    entry_cost = STARTING_CAPITAL * TRANSACTION_COST_PCT
    units = (STARTING_CAPITAL - entry_cost) / entry_price

    end_price = window["Close"].iloc[-1]
    end_value = units * end_price
    total_return = (end_value / STARTING_CAPITAL - 1) * 100

    return {"start_equity": STARTING_CAPITAL, "end_equity": end_value, "total_return": total_return}


def print_buy_and_hold(bh: dict, label: str):
    if bh is None:
        print(f"  {label}: not enough data")
        return
    print(f"  {label}:")
    print(f"    Start equity: ${bh['start_equity']:,.2f}  ->  End equity: ${bh['end_equity']:,.2f}")
    print(f"    Total return: {bh['total_return']:.2f}%  (1 trade, buy-and-hold, no exit)")


# Step 3: summarize performance (full period + holdout-only)

def summarize_performance(equity_df: pd.DataFrame, trades_df: pd.DataFrame, label: str):
    if len(equity_df) < 2:
        print(f"  {label}: not enough data")
        return

    start_val = equity_df["equity"].iloc[0]
    end_val = equity_df["equity"].iloc[-1]
    total_return = (end_val / start_val - 1) * 100

    n_trades = len(trades_df) if trades_df is not None else 0
    total_costs = trades_df["cost"].sum() if trades_df is not None and len(trades_df) else 0.0

    print(f"  {label}:")
    print(f"    Start equity: ${start_val:,.2f}  ->  End equity: ${end_val:,.2f}")
    print(f"    Total return: {total_return:.2f}%")
    print(f"    Trades: {n_trades}   Total transaction costs paid: ${total_costs:,.2f}")


# Main

def main():
    best_params = get_best_params_per_asset(OPTIMIZATION_RESULTS_PATH)
    out_dir = Path("data_processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    for asset_name, path in ASSETS.items():
        params = best_params[asset_name]
        ma_short, ma_long = params["ma_short"], params["ma_long"]

        df = pd.read_csv(path, index_col="Date", parse_dates=True).sort_index()
        equity_df, trades_df = run_backtest(df, ma_short, ma_long)

        equity_df.to_csv(out_dir / f"{asset_name}_equity_curve.csv")
        trades_df.to_csv(out_dir / f"{asset_name}_trades.csv", index=False)

        print(f"\n{asset_name.upper()} (ma_short={ma_short}, ma_long={ma_long})")
        print(" STRATEGY:")
        summarize_performance(equity_df, trades_df, "Full period (2016-present)")

        holdout_equity = equity_df.loc[equity_df.index >= HOLDOUT_START]
        holdout_trades = trades_df[trades_df["date"] >= HOLDOUT_START] if len(trades_df) else trades_df
        summarize_performance(holdout_equity, holdout_trades, f"Holdout only ({HOLDOUT_START}-present)")

        print(" BUY-AND-HOLD BENCHMARK:")
        bh_full = compute_buy_and_hold(df, df.index.min())
        print_buy_and_hold(bh_full, "Full period (2016-present)")

        bh_holdout = compute_buy_and_hold(df, HOLDOUT_START)
        print_buy_and_hold(bh_holdout, f"Holdout only ({HOLDOUT_START}-present)")

    print(f"\nSaved equity curves + trade logs to {out_dir}/")


if __name__ == "__main__":
    main()