import numpy as np
import pandas as pd


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # log returns
    df["stock_logret"] = np.log(df["stock_close"] / df["stock_close"].shift(1))
    df["market_logret"] = np.log(df["market_close"] / df["market_close"].shift(1))

    # rolling volatility
    df["stock_vol20"] = df["stock_logret"].rolling(20).std()
    df["market_vol20"] = df["market_logret"].rolling(20).std()

    # momentum (mean return)
    df["stock_mean5"] = df["stock_logret"].rolling(5).mean()
    df["stock_mean20"] = df["stock_logret"].rolling(20).mean()

    # relative strength vs market
    df["rel_logret"] = df["stock_logret"] - df["market_logret"]

    # volume change
    df["vol_chg"] = df["stock_volume"].pct_change()

    df.dropna(inplace=True)
    return df
