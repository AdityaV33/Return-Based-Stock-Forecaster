import numpy as np
import pandas as pd


def make_market_dataset(
    df_feat: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int = 60,
    horizon: int = 10
):
    """
    Returns:
    X_flat: (samples, seq_len * n_features)  for XGBoost
    y_cum: (samples,) cumulative return (simple return over horizon)
    y_trend: (samples,) 0/1 bullish label
    """
    X_list = []
    y_cum_list = []
    y_trend_list = []

    values = df_feat[feature_cols].values
    stock_ret = df_feat["stock_logret"].values

    for i in range(seq_len, len(df_feat) - horizon):
        x_seq = values[i - seq_len:i]
        future = stock_ret[i:i + horizon]

        # safety checks
        if not np.isfinite(x_seq).all():
            continue
        if not np.isfinite(future).all():
            continue

        log_sum = future.sum()

        # prevent overflow in exp
        log_sum = np.clip(log_sum, -0.5, 0.5)

        # simple cumulative return
        cum_ret = np.exp(log_sum) - 1

        # clip extreme moves
        cum_ret = np.clip(cum_ret, -0.20, 0.20)

        if not np.isfinite(cum_ret):
            continue

        trend = 1 if cum_ret > 0 else 0

        X_list.append(x_seq.flatten())
        y_cum_list.append(cum_ret)
        y_trend_list.append(trend)

    print("Samples created:", len(X_list))

    X_flat = np.array(X_list, dtype=np.float32)
    y_cum = np.array(y_cum_list, dtype=np.float32)
    y_trend = np.array(y_trend_list, dtype=np.int64)

    return X_flat, y_cum, y_trend
