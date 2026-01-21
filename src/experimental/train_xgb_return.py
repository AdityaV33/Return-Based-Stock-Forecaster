import os
import joblib
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_loader import get_stock_and_market_data
from features_market import add_market_features
from dataset_market import make_market_dataset

SEQ_LEN = 60
HORIZON = 10

MODEL_DIR = "models"


def safe_ticker_name(ticker: str) -> str:
    return ticker.replace(".", "_")


if __name__ == "__main__":
    ticker = input("Enter ticker to train RETURN (example: RELIANCE.NS / AAPL): ").strip()

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

    # time split
    n = len(X)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    X_train, y_train = X[:train_end], y_cum[:train_end]
    X_val, y_val = X[train_end:val_end], y_cum[train_end:val_end]
    X_test, y_test = X[val_end:], y_cum[val_end:]

    model = XGBRegressor(
        n_estimators=2000,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        objective="reg:squarederror",
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    baseline0 = np.zeros_like(y_test)
    mae0 = mean_absolute_error(y_test, baseline0)
    rmse0 = np.sqrt(mean_squared_error(y_test, baseline0))

    print(f"\nTicker: {ticker} | Market: {idx}")
    print("\n=== CUMULATIVE RETURN TEST METRICS ===")
    print(f"XGBRegressor -> MAE: {mae:.6f} | RMSE: {rmse:.6f}")
    print(f"Baseline(0)  -> MAE: {mae0:.6f} | RMSE: {rmse0:.6f}")

    corr = np.corrcoef(y_test, preds)[0, 1]
    print(f"Correlation(y, pred): {corr:.4f}")

    sign_acc = np.mean((preds > 0) == (y_test > 0))
    print("Sign Accuracy:", sign_acc)

    # save per ticker
    os.makedirs(MODEL_DIR, exist_ok=True)
    name = safe_ticker_name(ticker)

    ret_model_path = os.path.join(MODEL_DIR, f"{name}_return.pkl")
    cfg_path = os.path.join(MODEL_DIR, f"{name}_config.pkl")

    joblib.dump(model, ret_model_path)

    cfg = {
        "ticker": ticker,
        "market_index": idx,
        "seq_len": SEQ_LEN,
        "horizon": HORIZON,
        "feature_cols": feature_cols,
    }
    joblib.dump(cfg, cfg_path)

    print("\nSaved return model:", ret_model_path)
    print("Saved config:", cfg_path)
