"""
optimizer.py
----------------------------------------------------------
Reusable optimization engine for Walk-Forward Validation.

Responsibilities
----------------
1. Create expanding-window folds.
2. Evaluate MA parameter combinations.
3. Find the best MA pair on the training set.

No file reading.
No CSV writing.
No printing.

This module is intentionally reusable.
"""

from itertools import product
import numpy as np
import pandas as pd

from signals import build_signals


# ==========================================================
# Search Space
# ==========================================================

SHORT_WINDOWS = [10, 15, 20, 30, 40]
LONG_WINDOWS = [50, 80, 100, 120, 150]

RSI_PERIOD = 14
OVERSOLD = 30
OVERBOUGHT = 70

TRADING_DAYS = 252


# ==========================================================
# Walk Forward Fold Generator
# ==========================================================

def create_walkforward_folds(
    first_train_year=2016,
    first_test_year=2020,
    last_test_year=2024,
):
    """
    Creates expanding-window folds.

    Example

    Train : 2016-2019
    Test  : 2020

    Train : 2016-2020
    Test  : 2021

    ...
    """

    folds = []

    for test_year in range(first_test_year, last_test_year + 1):

        folds.append({

            "train_start": f"{first_train_year}-01-01",
            "train_end": f"{test_year-1}-12-31",

            "test_start": f"{test_year}-01-01",
            "test_end": f"{test_year}-12-31",

            "test_year": test_year

        })

    return folds


# ==========================================================
# Performance Evaluation
# ==========================================================

def evaluate_strategy(df, ma_short, ma_long):
    """
    Evaluates one MA parameter combination.

    Returns
    -------
    dict
    {
        "sharpe": ...,
        "total_return": ...,
        "num_trades": ...
    }
    """

    signals = build_signals(
        df,
        ma_short,
        ma_long,
        RSI_PERIOD,
        OVERSOLD,
        OVERBOUGHT
    )

    # Daily strategy returns
    returns = (
        signals["Close"]
        .pct_change()
        .fillna(0)
        * signals["position"]
    )

    if len(returns) < 2:
        return None

    volatility = returns.std()

    if volatility == 0 or np.isnan(volatility):
        return None

    sharpe = (
        returns.mean()
        / volatility
        * np.sqrt(TRADING_DAYS)
    )

    # Compound total return
    total_return = ((1 + returns).prod() - 1) * 100

    # Count completed trades (BUY + SELL)
    num_trades = int((signals["signal"].abs() == 1).sum() // 2)

    return {
        "sharpe": sharpe,
        "total_return": total_return,
        "num_trades": num_trades,
    }


# ==========================================================
# Grid Search
# ==========================================================

def find_best_parameters(
    train_df,
    short_windows=SHORT_WINDOWS,
    long_windows=LONG_WINDOWS,
):
    """
    Finds the best MA combination.

    Selection Rule
    --------------
    1. Ignore strategies with fewer than 3 trades.
    2. Highest Sharpe wins.
    """

    best = None
    best_score = -np.inf

    for ma_short, ma_long in product(short_windows, long_windows):

        if ma_short >= ma_long:
            continue

        metrics = evaluate_strategy(
            train_df,
            ma_short,
            ma_long,
        )

        if metrics is None:
            continue

        # Ignore strategies that barely trade
        if metrics["num_trades"] < 3:
            continue

        if metrics["sharpe"] > best_score:

            best_score = metrics["sharpe"]

            best = {
                "ma_short": ma_short,
                "ma_long": ma_long,
                "train_sharpe": metrics["sharpe"],
                "train_return": metrics["total_return"],
                "num_trades": metrics["num_trades"],
            }

    return best