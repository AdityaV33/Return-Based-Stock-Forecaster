import numpy as np
from xgboost import XGBClassifier, XGBRegressor

from data_loader import get_stock_and_market_data
from features_market import add_market_features
from dataset_market import make_market_dataset

print("RUN_FORECAST STARTED")


SEQ_LEN = 60
HORIZON = 10

BULL_TH = 0.55
BEAR_TH = 0.45


def train_models_for_ticker(ticker: str):
    df, idx = get_stock_and_market_data(ticker, start="2018-01-01", end="2025-01-01")
    df_feat = add_market_features(df)

    feature_cols = [
        "stock_close",
        "stock_volume",
        "market_close",
        "stock_logret",
        "market_logret",
        "stock_vol20",
        "market_vol20",
        "stock_mean5",
        "stock_mean20",
        "rel_logret",
        "vol_chg",
    ]

    X, y_cum, y_trend = make_market_dataset(
        df_feat,
        feature_cols=feature_cols,
        seq_len=SEQ_LEN,
        horizon=HORIZON
    )

    # time split
    n = len(X)
    train_end = int(0.85 * n)

    X_train, ytrend_train = X[:train_end], y_trend[:train_end]
    X_train2, ycum_train = X[:train_end], y_cum[:train_end]

    # class weighting for bullish recall
    pos = int((ytrend_train == 1).sum())
    neg = int((ytrend_train == 0).sum())
    scale_pos_weight = neg / max(pos, 1)

    clf = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight
    )
    clf.fit(X_train, ytrend_train)

    reg = XGBRegressor(
        objective="reg:pseudohubererror",
        n_estimators=800,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
    )
    reg.fit(X_train2, ycum_train)

    return clf, reg, df_feat, idx, feature_cols


def forecast_next_n_days(ticker: str):
    clf, reg, df_feat, idx, feature_cols = train_models_for_ticker(ticker)

    # latest window
    values = df_feat[feature_cols].values
    last_seq = values[-SEQ_LEN:]
    x_input = last_seq.flatten().reshape(1, -1)

    trend_prob = float(clf.predict_proba(x_input)[0, 1])
    exp_return = float(reg.predict(x_input)[0])

    if trend_prob >= BULL_TH:
        decision = "Bullish 📈"
    elif trend_prob <= BEAR_TH:
        decision = "Bearish 📉"
    else:
        decision = "Neutral ⚖️ (No-trade zone)"

    last_close = float(df_feat["stock_close"].iloc[-1])
    predicted_price = last_close * (1 + exp_return)

    print("\n==============================")
    print("Ticker:", ticker)
    print("Market index:", idx)
    print("Horizon (trading days):", HORIZON)
    print("------------------------------")
    print("Trend probability (Bullish):", round(trend_prob, 4))
    print("Expected N-day return:", round(exp_return * 100, 3), "%")
    print("Decision:", decision)
    print("------------------------------")
    print("Last Close:", round(last_close, 2))
    print("Predicted Price (rough):", round(predicted_price, 2))
    print("==============================\n")


if __name__ == "__main__":
    ticker = input("Enter ticker (example: RELIANCE.NS / AAPL): ").strip()
    forecast_next_n_days(ticker)
