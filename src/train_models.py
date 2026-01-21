import os
import joblib
import numpy as np
from xgboost import XGBClassifier, XGBRegressor

from data_loader import get_stock_and_market_data
from features_market import add_market_features
from dataset_market import make_market_dataset

SEQ_LEN = 60
HORIZON = 10
MODELS_DIR = "models"


def safe_ticker_name(ticker: str) -> str:
    return ticker.replace(".", "_")


def train_and_save(ticker: str):
    os.makedirs(MODELS_DIR, exist_ok=True)

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

    print("Samples created:", len(X))

    # time split (train only)
    n = len(X)
    train_end = int(0.85 * n)

    X_train = X[:train_end]
    ytrend_train = y_trend[:train_end]
    ycum_train = y_cum[:train_end]

    # imbalance handling for trend
    pos = int((ytrend_train == 1).sum())
    neg = int((ytrend_train == 0).sum())
    scale_pos_weight = neg / max(pos, 1)

    print("Train class counts -> Neg:", neg, "| Pos:", pos)
    print("scale_pos_weight:", scale_pos_weight)

    # -------------------------
    # Trend classifier
    # -------------------------
    clf = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
    )
    clf.fit(X_train, ytrend_train)

    # -------------------------
    # Return regressor
    # -------------------------
    reg = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=800,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
    )
    reg.fit(X_train, ycum_train)

    # -------------------------
    # Save models (PKL)
    # -------------------------
    name = safe_ticker_name(ticker)

    trend_path = os.path.join(MODELS_DIR, f"{name}_trend.pkl")
    ret_path = os.path.join(MODELS_DIR, f"{name}_return.pkl")
    cfg_path = os.path.join(MODELS_DIR, f"{name}_config.pkl")
    th_path = os.path.join(MODELS_DIR, f"{name}_threshold.txt")

    joblib.dump(clf, trend_path)
    joblib.dump(reg, ret_path)

    config = {
        "ticker": ticker,
        "market_index": idx,
        "seq_len": SEQ_LEN,
        "horizon": HORIZON,
        "feature_cols": feature_cols,
    }
    joblib.dump(config, cfg_path)

    # Save threshold (optional — predict.py expects it)
    # We keep 0.5 for now since you use global thresholds anyway.
    with open(th_path, "w") as f:
        f.write("0.5")

    print("\nSaved:")
    print("Trend model :", trend_path)
    print("Return model:", ret_path)
    print("Config     :", cfg_path)
    print("Threshold  :", th_path)


if __name__ == "__main__":
    ticker = input("Enter ticker to train (example: RELIANCE.NS / AAPL): ").strip()
    train_and_save(ticker)
