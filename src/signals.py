"""Reusable, parameterized signal logic (MA trend filter + RSI entry/exit)."""

import pandas as pd


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def build_signals(df: pd.DataFrame, ma_short: int, ma_long: int,
                  rsi_period: int = 14, oversold: int = 30, overbought: int = 70) -> pd.DataFrame:
    df = df.copy()

    df["ma_short"] = df["Close"].rolling(window=ma_short).mean()
    df["ma_long"] = df["Close"].rolling(window=ma_long).mean()
    df["rsi"] = compute_rsi(df["Close"], rsi_period)

    trend_up = df["ma_short"] > df["ma_long"]
    rsi_recovering = (df["rsi"] > oversold) & (df["rsi"].shift(1) <= oversold)
    trend_flip_down = (df["ma_short"] < df["ma_long"]) & (df["ma_short"].shift(1) >= df["ma_long"].shift(1))
    rsi_overbought = (df["rsi"] > overbought) & (df["rsi"].shift(1) <= overbought)

    entry_signal = trend_up & rsi_recovering
    exit_signal = trend_flip_down | rsi_overbought

    position = [0] * len(df)
    in_position = False
    for i in range(len(df)):
        if not in_position and entry_signal.iloc[i]:
            in_position = True
        elif in_position and exit_signal.iloc[i]:
            in_position = False
        position[i] = 1 if in_position else 0

    # --- THE FIX: Aligning index and shifting to prevent time-travel ---
    df["position"] = position                                
    df["position"] = df["position"].shift(1).fillna(0)       
    df["signal"] = df["position"].diff().fillna(0)           

    return df