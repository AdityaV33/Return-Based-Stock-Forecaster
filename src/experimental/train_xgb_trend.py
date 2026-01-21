import os
import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, f1_score

from data_loader import get_stock_and_market_data
from features_market import add_market_features
from dataset_market import make_market_dataset

SEQ_LEN = 60
HORIZON = 10

MODEL_DIR = "models"


def safe_ticker_name(ticker: str) -> str:
    return ticker.replace(".", "_")


if __name__ == "__main__":
    ticker = input("Enter ticker to train TREND (example: RELIANCE.NS / AAPL): ").strip()

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

    X_train, y_train = X[:train_end], y_trend[:train_end]
    X_val, y_val = X[train_end:val_end], y_trend[train_end:val_end]
    X_test, y_test = X[val_end:], y_trend[val_end:]

    # imbalance handling
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / max(pos, 1)

    print("Train class counts -> Neg:", neg, "| Pos:", pos)
    print("scale_pos_weight:", scale_pos_weight)

    model = XGBClassifier(
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

    model.fit(X_train, y_train)

    # tune threshold on VAL
    val_probs = model.predict_proba(X_val)[:, 1]

    best_t, best_f1 = None, -1
    for t in np.linspace(0.05, 0.95, 19):
        val_preds = (val_probs >= t).astype(int)
        f1 = f1_score(y_val, val_preds)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    print("\nBest threshold on VAL:", best_t)
    print("Best VAL F1:", best_f1)

    # test evaluation using best_t
    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= best_t).astype(int)

    print("\nTicker:", ticker, "| Market:", idx)
    print("Test AUC:", roc_auc_score(y_test, test_probs))
    print("\nClassification Report (TEST):")
    print(classification_report(y_test, test_preds, digits=4))
    print("Confusion Matrix (TEST):")
    print(confusion_matrix(y_test, test_preds))

    # save per ticker
    os.makedirs(MODEL_DIR, exist_ok=True)
    name = safe_ticker_name(ticker)

    trend_model_path = os.path.join(MODEL_DIR, f"{name}_trend.pkl")
    cfg_path = os.path.join(MODEL_DIR, f"{name}_config.pkl")
    th_path = os.path.join(MODEL_DIR, f"{name}_threshold.txt")

    joblib.dump(model, trend_model_path)

    cfg = {
        "ticker": ticker,
        "market_index": idx,
        "seq_len": SEQ_LEN,
        "horizon": HORIZON,
        "feature_cols": feature_cols,
    }
    joblib.dump(cfg, cfg_path)

    with open(th_path, "w") as f:
        f.write(str(best_t))

    print("\nSaved trend model:", trend_model_path)
    print("Saved config:", cfg_path)
    print("Saved threshold:", th_path)
