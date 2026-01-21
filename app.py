import streamlit as st
import subprocess
import sys
import os

st.set_page_config(page_title="Return-Based Stock Forecaster", layout="centered")

st.title("📈 Return-Based Stock Forecaster")
st.caption("Predict next N-day return + bullish/bearish probability using XGBoost + market context")

ticker = st.text_input("Enter Stock Ticker", value="AAPL").strip()

col1, col2 = st.columns(2)
train_btn = col1.button("🧠 Train Models")
predict_btn = col2.button("📊 Predict")

st.divider()


def safe_name(ticker: str) -> str:
    return ticker.replace(".", "_")


def models_exist(ticker: str) -> bool:
    name = safe_name(ticker)
    trend_path = os.path.join("models", f"{name}_trend.pkl")
    ret_path = os.path.join("models", f"{name}_return.pkl")
    cfg_path = os.path.join("models", f"{name}_config.pkl")
    th_path = os.path.join("models", f"{name}_threshold.txt")
    return all(os.path.exists(p) for p in [trend_path, ret_path, cfg_path, th_path])


def run_script(script_name: str, ticker: str):
    """Run a python script using the current interpreter."""
    result = subprocess.run(
        [sys.executable, f"src/{script_name}"],
        input=ticker + "\n",
        text=True,
        capture_output=True
    )
    return result.stdout, result.stderr


if train_btn:
    if ticker == "":
        st.warning("⚠️ Please enter a valid ticker.")
    else:
        st.info("Training models... this may take 1–2 minutes ⏳")
        out, err = run_script("train_models.py", ticker)

        if err and "YFRateLimitError" in err:
            st.error("⚠️ Yahoo Finance rate-limited you. Please try again in 30–60 seconds.")
        elif err:
            st.error("⚠️ Training failed. Please check the ticker or try again later.")
            st.text(err)

        st.text(out)

if predict_btn:
    if ticker == "":
        st.warning("⚠️ Please enter a valid ticker.")
    elif not models_exist(ticker):
        st.warning("⚠️ No trained model found. Please click **Train Models** first.")
    else:
        st.info("Generating forecast...")
        out, err = run_script("predict.py", ticker)

        if err and "UnicodeEncodeError" in err:
            st.warning("⚠️ Emoji printing issue on Windows terminal encoding. Removing emojis fixes it.")
        elif err and "YFRateLimitError" in err:
            st.error("⚠️ Yahoo Finance rate-limited you. Try again after 30–60 seconds.")
        elif err:
            st.error("⚠️ Prediction failed. Try retraining or using another ticker.")
            st.text(err)

        st.text(out)
