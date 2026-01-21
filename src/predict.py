import os
import joblib
import numpy as np

from data_loader import get_stock_and_market_data
from features_market import add_market_features

MODELS_DIR = "models"

# Global decision thresholds
BULL_TH = 0.55
BEAR_TH = 0.45


def safe_ticker_name(ticker: str) -> str:
    return ticker.replace(".", "_")


def load_models_for_ticker(ticker: str):
    name = safe_ticker_name(ticker)

    trend_path = os.path.join(MODELS_DIR, f"{name}_trend.pkl")
    ret_path = os.path.join(MODELS_DIR, f"{name}_return.pkl")
    cfg_path = os.path.join(MODELS_DIR, f"{name}_config.pkl")

    if not os.path.exists(trend_path):
        raise FileNotFoundError(f"Missing trend model: {trend_path} (run train_models.py)")
    if not os.path.exists(ret_path):
        raise FileNotFoundError(f"Missing return model: {ret_path} (run train_models.py)")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing config: {cfg_path} (run train_models.py)")

    clf = joblib.load(trend_path)
    reg = joblib.load(ret_path)
    cfg = joblib.load(cfg_path)

    return clf, reg, cfg


def predict_next_n_days(ticker: str, start="2018-01-01", end="2025-01-01"):
    clf, reg, cfg = load_models_for_ticker(ticker)

    df, idx = get_stock_and_market_data(ticker, start=start, end=end)
    df_feat = add_market_features(df)

    seq_len = cfg["seq_len"]
    horizon = cfg["horizon"]
    feature_cols = cfg["feature_cols"]

    values = df_feat[feature_cols].values
    if len(values) < seq_len:
        raise ValueError(f"Not enough data. Need {seq_len} rows, got {len(values)}")

    last_seq = values[-seq_len:]
    x_input = last_seq.flatten().reshape(1, -1)

    trend_prob = float(clf.predict_proba(x_input)[0, 1])
    exp_logret_sum = float(reg.predict(x_input)[0])

    # Decision logic
    if trend_prob >= BULL_TH:
        decision = "Bullish "
    elif trend_prob <= BEAR_TH:
        decision = "Bearish "
    else:
        decision = "Neutral  (No-trade zone)"

    last_close = float(df_feat["stock_close"].iloc[-1])

    pct_return = (np.exp(exp_logret_sum) - 1) * 100
    predicted_price = last_close * np.exp(exp_logret_sum)

    print("\n==============================")
    print("Ticker:", ticker)
    print("Market index:", idx)
    print("Horizon (trading days):", horizon)
    print("------------------------------")
    print("Trend probability (Bullish):", round(trend_prob, 4))
    print("Expected N-day return:", round(pct_return, 3), "%")
    print("Decision:", decision)
    print("------------------------------")
    print("Last Close:", round(last_close, 2))
    print("Predicted Price (rough):", round(predicted_price, 2))
    print("==============================\n")


if __name__ == "__main__":
    ticker = input("Enter ticker to predict (example: RELIANCE.NS / AAPL): ").strip()
    predict_next_n_days(ticker)
