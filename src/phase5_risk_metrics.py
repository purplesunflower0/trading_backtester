"""
PHASE 5 — Risk Metrics
Calculates Sharpe ratio, volatility, and maximum drawdown to evaluate the quality of returns.
"""

import pandas as pd
import numpy as np
import os

INPUT_DIR = "data_backtested"

def calculate_drawdown(equity_series: pd.Series) -> float:
    """Calculates the maximum percentage drop from a peak."""
    rolling_max = equity_series.cummax()
    drawdowns = (equity_series - rolling_max) / rolling_max
    return drawdowns.min() * 100  # Return as a percentage

def analyze_risk(df: pd.DataFrame, asset_name: str):
    """Computes risk metrics for both the strategy and the benchmark."""
    
    # We use 252 trading days in a year to annualize our daily metrics
    TRADING_DAYS = 252 
    
    # --- Strategy Metrics ---
    strat_daily_returns = df["net_return"]
    strat_annual_return = (df["equity"].iloc[-1] / df["equity"].iloc[0]) ** (TRADING_DAYS / len(df)) - 1
    strat_volatility = strat_daily_returns.std() * np.sqrt(TRADING_DAYS)
    
    # Assuming Risk-Free Rate is ~0% for simplicity
    strat_sharpe = strat_annual_return / strat_volatility if strat_volatility != 0 else 0
    strat_max_dd = calculate_drawdown(df["equity"])

    # --- Buy & Hold (Benchmark) Metrics ---
    bh_daily_returns = df["asset_return"]
    bh_annual_return = (df["buy_hold_equity"].iloc[-1] / df["buy_hold_equity"].iloc[0]) ** (TRADING_DAYS / len(df)) - 1
    bh_volatility = bh_daily_returns.std() * np.sqrt(TRADING_DAYS)
    
    bh_sharpe = bh_annual_return / bh_volatility if bh_volatility != 0 else 0
    bh_max_dd = calculate_drawdown(df["buy_hold_equity"])

    print(f"--- Risk Analysis: {asset_name.upper()} ---")
    print(f"{'Metric':<20} | {'Strategy':<15} | {'Buy & Hold (Benchmark)':<15}")
    print("-" * 55)
    print(f"{'Annualized Return':<20} | {strat_annual_return*100:>14.2f}% | {bh_annual_return*100:>14.2f}%")
    print(f"{'Annual Volatility':<20} | {strat_volatility*100:>14.2f}% | {bh_volatility*100:>14.2f}%")
    print(f"{'Sharpe Ratio':<20} | {strat_sharpe:>14.2f}  | {bh_sharpe:>14.2f}")
    print(f"{'Max Drawdown':<20} | {strat_max_dd:>14.2f}% | {bh_max_dd:>14.2f}%\n")

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"Error: {INPUT_DIR} not found. Run Phase 4 first.")
        return

    print("Executing Phase 5: Risk Metric Calculation\n")
    
    for filename in os.listdir(INPUT_DIR):
        if not filename.endswith("_backtest.csv"):
            continue

        name = filename.replace("_backtest.csv", "")
        path = os.path.join(INPUT_DIR, filename)

        df = pd.read_csv(path, index_col="Date", parse_dates=True)
        analyze_risk(df, name)

if __name__ == "__main__":
    main()