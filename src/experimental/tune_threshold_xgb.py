import numpy as np
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from data_loader import get_stock_and_market_data
from features_market import add_market_features
from dataset_market import make_market_dataset

SEQ_LEN = 60
HORIZON = 10

ticker = "RELIANCE.NS"

df, idx = get_stock_and_market_data(ticker, start="2018-01-01", end="2025-01-01")
df_feat = add_market_features(df)

feature_cols = [
    "stock_close","stock_volume","market_close",
    "stock_logret","market_logret",
    "stock_vol20","market_vol20",
    "stock_mean5","stock_mean20",
    "rel_logret","vol_chg"
]

X, y_cum, y = make_market_dataset(df_feat, feature_cols, seq_len=SEQ_LEN, horizon=HORIZON)

n = len(X)
train_end = int(0.7 * n)
val_end = int(0.85 * n)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    eval_metric="logloss"
)
model.fit(X_train, y_train)

val_probs = model.predict_proba(X_val)[:, 1]

best_t, best_f1 = None, -1
for t in np.linspace(0.05, 0.95, 19):
    preds = (val_probs >= t).astype(int)
    f1 = f1_score(y_val, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_t = t

print("Best threshold on VAL:", best_t)
print("Best F1 on VAL:", best_f1)
