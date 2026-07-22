import yfinance as yf
import pandas as pd
import os
import datetime

TICKERS = {
    "GC=F": "gold",
    "CL=F": "crude",
    "^GSPC": "sp500",
}

START_DATE = "2016-01-01"

END_DATE = datetime.date.today().strftime("%Y-%m-%d")
  

OUTPUT_DIR = "data_raw"


def pull_and_clean(ticker: str, name: str) -> pd.DataFrame:
    """Download one ticker's daily OHLCV and do basic cleaning."""

    print(f"Downloading {ticker} ({name}) ...")

    df = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        progress=False,
        auto_adjust=True,
    )

    if df.empty:
        raise ValueError(f"No data returned for {ticker} — check the ticker symbol.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # --- FIX: guard against real-world price anomalies (e.g. WTI crude going
    # negative on April 20, 2020 during the COVID demand collapse). A near-zero
    # or negative price breaks pct_change()-based return calculations downstream,
    # so we floor all price columns at a small positive value.
    price_cols = ["Open", "High", "Low", "Close"]
    df[price_cols] = df[price_cols].clip(lower=0.01)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    n_missing = df.isna().sum().sum()
    if n_missing > 0:
        print(f"  -> found {n_missing} missing cell(s), forward-filling")
        df = df.ffill()

    df = df.dropna()

    df.index.name = "Date"
    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for ticker, name in TICKERS.items():
        df = pull_and_clean(ticker, name)
        out_path = os.path.join(OUTPUT_DIR, f"{name}.csv")
        df.to_csv(out_path)
        print(f"  -> saved {len(df)} rows to {out_path}")
        print(f"  -> date range: {df.index.min().date()} to {df.index.max().date()}")
        print()


if __name__ == "__main__":
    main()