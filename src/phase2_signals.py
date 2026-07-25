"""
PHASE 2

Strategy logic:
  - Trend filter: 20-day MA > 50-day MA  => uptrend
  - Entry (go long): uptrend AND RSI crosses back above 30 (recovering from oversold)
  - Exit (go flat):  trend flips down  OR RSI crosses above 70 (overbought)
"""

import pandas as pd
import os

INPUT_DIR = "data_raw"
OUTPUT_DIR = "data_processed"

SHORT_WINDOW = 20
LONG_WINDOW = 50
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70


def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ma_short"] = df["Close"].rolling(window=SHORT_WINDOW).mean()
    df["ma_long"] = df["Close"].rolling(window=LONG_WINDOW).mean()
    df["rsi"] = compute_rsi(df["Close"])

    df["trend_up"] = df["ma_short"] > df["ma_long"]

    # RSI crossing back above 30 = recovering from oversold
    rsi_recovering = (df["rsi"] > RSI_OVERSOLD) & (df["rsi"].shift(1) <= RSI_OVERSOLD)

    # Trend flipping down = short MA crosses below long MA
    trend_flip_down = (df["ma_short"] < df["ma_long"]) & (df["ma_short"].shift(1) >= df["ma_long"].shift(1))

    # RSI crossing above 70 = overbought
    rsi_overbought = (df["rsi"] > RSI_OVERBOUGHT) & (df["rsi"].shift(1) <= RSI_OVERBOUGHT)

    entry_signal = df["trend_up"] & rsi_recovering
    exit_signal = trend_flip_down | rsi_overbought

    # State machine: walk day by day and decide if we're in or out.
    position = [0] * len(df)
    in_position = False
    for i in range(len(df)):
        if not in_position and entry_signal.iloc[i]:
            in_position = True
        elif in_position and exit_signal.iloc[i]:
            in_position = False
        position[i] = 1 if in_position else 0
    
    # --- THE FIX: Aligning index and shifting to prevent time-travel ---
    df["position"] = position                                # 1. Assign list to inherit Date index
    df["position"] = df["position"].shift(1).fillna(0)       # 2. Shift to fix look-ahead bias
    df["signal"] = df["position"].diff().fillna(0)           # 3. Calculate safe entry/exit signals

    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for filename in os.listdir(INPUT_DIR):
        if not filename.endswith(".csv"):
            continue

        name = filename.replace(".csv", "")
        path = os.path.join(INPUT_DIR, filename)

        df = pd.read_csv(path, index_col="Date", parse_dates=True)
        df = build_signals(df)

        out_path = os.path.join(OUTPUT_DIR, f"{name}_signals.csv")
        df.to_csv(out_path)

        n_trades = (df["signal"] == 1).sum()
        print(f"{name}: {n_trades} entry signals generated -> saved to {out_path}")


if __name__ == "__main__":
    main()