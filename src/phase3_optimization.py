"""
PHASE 3Parameter Optimization (Grid Search with Walk-Forward Validation)
"""

import pandas as pd
import itertools
import os

# Import your patched, look-ahead-free signal function
from signals import build_signals

INPUT_FILE = "data_raw/sp500.csv"

def evaluate_returns(df):
    """Calculates a rough cumulative return for optimization sorting."""
    
    # Calculate daily percent change of the asset
    df["pct_change"] = df["Close"].pct_change().fillna(0)
    
    # Strategy return = Daily change * our position (1 or 0)
    df["strategy_return"] = df["position"] * df["pct_change"]
    
    # Return as a percentage
    return df["strategy_return"].sum() * 100 

def main():
    print("--- PHASE 3: Grid Search Optimization ---")
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run Phase 1 first.")
        return

    df = pd.read_csv(INPUT_FILE, index_col="Date", parse_dates=True)

    # 1. Train/Test Split (Crucial for quantitative research)
    train_df = df.loc["2015":"2022"].copy()
    test_df = df.loc["2023":"2025"].copy()

    # 2. Define the hyperparameter grid
    short_ma_options = [10, 20, 30]
    long_ma_options = [50, 100, 150]
    
    best_return = -float('inf')
    best_params = None

    print("\nOptimizing on Training Data (2015-2022)...")
    
    # 3. The Grid Search
    for short, long in itertools.product(short_ma_options, long_ma_options):
        if short >= long:
            continue # Skip illogical pairs where short MA is longer than long MA
            
        # Run our signal builder
        tested_df = build_signals(train_df, ma_short=short, ma_long=long)
        strat_return = evaluate_returns(tested_df)
        
        print(f"Tested {short}/{long} MA -> Return: {strat_return:.2f}%")
        
        if strat_return > best_return:
            best_return = strat_return
            best_params = (short, long)

    print(f"\n[WINNER] Best Parameters: {best_params[0]} Short / {best_params[1]} Long")
    print(f"Training Data Return: {best_return:.2f}%")

    # 4. Out-of-Sample Validation ("I didn't get lucky" test)
    print("\nValidating on Unseen Test Data (2023-2025)...")
    validation_df = build_signals(test_df, ma_short=best_params[0], ma_long=best_params[1])
    val_return = evaluate_returns(validation_df)
    
    print(f"Test Data Return: {val_return:.2f}%")
    if val_return > 0:
        print("Verdict: The strategy held up on unseen data. Ready for Phase 4!")
    else:
        print("Verdict: The strategy fell apart on unseen data (Overfitted).")

if __name__ == "__main__":
    main()