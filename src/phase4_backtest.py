"""
PHASE 4 — The Backtest Engine
Simulates trading the signals with real capital and transaction costs.
"""

import pandas as pd
import os
import numpy as np

INPUT_DIR = "data_processed"
OUTPUT_DIR = "data_backtested"

# --- Simulation Parameters ---
INITIAL_CAPITAL = 10000.0  # Start with $10,000
TRANSACTION_FEE = 0.001    # 0.1% fee per trade (Broker slippage/commission)

def run_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """Runs a vectorized backtest on a dataframe containing 'Close', 'position', and 'signal'."""
    df = df.copy()

    # 1. Calculate the daily percentage change of the asset itself
    df["asset_return"] = df["Close"].pct_change().fillna(0)
    df["asset_return"] = df["asset_return"].clip(lower=-0.5, upper=0.5)

    # 2. Calculate raw strategy return 
    # (Since we fixed the look-ahead bias in Phase 2, 'position' is safe to multiply directly)
    df["strat_return"] = df["position"] * df["asset_return"]

    # 3. Apply Transaction Costs
    # We only pay a fee on days where the signal is +1 (Buy) or -1 (Sell/Close)
    df["trade_cost"] = df["signal"].abs() * TRANSACTION_FEE
    
    # 4. Net Daily Return
    df["net_return"] = df["strat_return"] - df["trade_cost"]

    # 5. Calculate the Equity Curve (Compounding the returns)
    # cumprod() efficiently multiplies the daily growth factors: (1 + r1) * (1 + r2) * ...
    df["equity"] = INITIAL_CAPITAL * (1 + df["net_return"]).cumprod()

    # Calculate Buy & Hold equity for a benchmark comparison
    df["buy_hold_equity"] = INITIAL_CAPITAL * (1 + df["asset_return"]).cumprod()

    return df

def calculate_trade_metrics(df: pd.DataFrame) -> dict:
    """Calculates win rate and total trades to evaluate the model's behavior."""
    # A trade is "won" if the price when we sell is higher than when we bought
    # This is a simplified proxy for win rate based on daily net returns while holding
    
    # Isolate days where we are holding a position
    holding_days = df[df["position"] == 1]
    
    if len(holding_days) == 0:
        return {"total_trades": 0, "win_rate": 0.0, "final_equity": INITIAL_CAPITAL}

    total_trades = int(df["signal"].abs().sum() / 2) # Div by 2 because 1 entry + 1 exit = 1 full trade
    
    # Calculate final portfolio value
    final_equity = df["equity"].iloc[-1]
    
    return {
        "total_trades": total_trades,
        "final_equity": final_equity
    }

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("--- PHASE 4: Executing Backtest Engine ---")

    for filename in os.listdir(INPUT_DIR):
        if not filename.endswith("_signals.csv"):
            continue

        name = filename.replace("_signals.csv", "")
        path = os.path.join(INPUT_DIR, filename)

        # Load the signals
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
        
        # Run the simulation
        backtested_df = run_backtest(df)

        # Save the results for Phase 5 (Risk) and Phase 6 (Visualization)
        out_path = os.path.join(OUTPUT_DIR, f"{name}_backtest.csv")
        backtested_df.to_csv(out_path)

        # Print a quick summary
        metrics = calculate_trade_metrics(backtested_df)
        bh_final = backtested_df["buy_hold_equity"].iloc[-1]
        
        print(f"\nAsset: {name.upper()}")
        print(f"  Total Trades Executed : {metrics['total_trades']}")
        print(f"  Strategy Final Equity : ${metrics['final_equity']:,.2f}")
        print(f"  Buy & Hold Equity     : ${bh_final:,.2f}")

if __name__ == "__main__":
    main()