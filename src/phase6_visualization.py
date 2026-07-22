"""
PHASE 6 — Visualization Engine
Generates comparison charts for Equity Curves and Drawdown Profiles.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

INPUT_DIR = "data_backtested"
OUTPUT_DIR = "charts"

def generate_charts(df: pd.DataFrame, asset_name: str):
    """Generates a two-panel chart showing equity curves and drawdown zones."""
    
    # 1. Compute daily drawdown percentages for both series
    strat_peaks = df["equity"].cummax()
    strat_dd = (df["equity"] - strat_peaks) / strat_peaks * 100
    
    bh_peaks = df["buy_hold_equity"].cummax()
    bh_dd = (df["buy_hold_equity"] - bh_peaks) / bh_peaks * 100

    # 2. Setup a two-panel plot (Top: Equity Curve, Bottom: Drawdown)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, 
        gridspec_kw={'height_ratios': [2, 1]}
    )

    # Top Panel: Equity Growth Simulation
    ax1.plot(df.index, df["equity"], label="Strategy Portfolio", linewidth=2)
    ax1.plot(df.index, df["buy_hold_equity"], label="Buy & Hold Benchmark", linestyle="--", linewidth=1.5)
    ax1.set_title(f"{asset_name.upper()} Performance & Risk Profile (2015 - 2025)", fontsize=14, pad=15)
    ax1.set_ylabel("Portfolio Value ($)", fontsize=11)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Bottom Panel: Underwater Drawdown Zones
    ax2.fill_between(df.index, strat_dd, 0, label="Strategy Drawdown", alpha=0.4)
    ax2.fill_between(df.index, bh_dd, 0, label="Benchmark Drawdown", alpha=0.2)
    ax2.set_ylabel("Drawdown (%)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    # Clean layout boundaries
    plt.tight_layout()
    
    # Save the asset chart
    out_path = os.path.join(OUTPUT_DIR, f"{asset_name}_backtest_results.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Successfully generated visual plot: {out_path}")

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"Error: {INPUT_DIR} folder not found. Please execute Phase 4 first.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("--- PHASE 6: Generating Performance Visualizations ---\n")

    for filename in os.listdir(INPUT_DIR):
        if not filename.endswith("_backtest.csv"):
            continue

        name = filename.replace("_backtest.csv", "")
        path = os.path.join(INPUT_DIR, filename)

        # Load historical backtest runs
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
        generate_charts(df, name)

if __name__ == "__main__":
    main()