import os
import time
import pandas as pd
import yfinance as yf
from market_context import get_market_index_for_ticker

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


def fetch_yfinance(ticker: str, start: str, end: str, retries: int = 3) -> pd.DataFrame:
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval="1d",
                progress=False,
                threads=False,
                auto_adjust=False
            )

            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df = df.dropna()
                df.index = pd.to_datetime(df.index)
                df.sort_index(inplace=True)
                return df

        except Exception as e:
            print(f"[yfinance Attempt {attempt}] Error: {e}")

        time.sleep(2)

    return pd.DataFrame()


def get_stock_data(ticker: str, start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    os.makedirs(DATA_DIR, exist_ok=True)
    safe_name = ticker.replace(".", "_")
    cache_path = os.path.join(DATA_DIR, f"{safe_name}_{start}_{end}.csv")

    if use_cache and os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        df.sort_index(inplace=True)
        return df

    df = fetch_yfinance(ticker, start, end)

    if df.empty:
        raise RuntimeError(f"Could not fetch data for ticker: {ticker}")

    df.index.name = "Date"
    df.to_csv(cache_path)
    return df


def get_stock_and_market_data(ticker: str, start="2018-01-01", end="2025-01-01"):
    market_index = get_market_index_for_ticker(ticker)

    stock = get_stock_data(ticker, start, end, use_cache=True)
    market = get_stock_data(market_index, start, end, use_cache=True)

    stock = stock[["Close", "Volume"]].rename(
        columns={"Close": "stock_close", "Volume": "stock_volume"}
    )
    market = market[["Close"]].rename(columns={"Close": "market_close"})

    df = stock.join(market, how="inner")
    df.dropna(inplace=True)

    return df, market_index
