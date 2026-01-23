# 📈 Return-Based Stock Forecaster
(Done by Aditya Verma and Daniel Newton Manogaran)
A **Streamlit** application that trains **XGBoost models** to forecast the **next N-day expected return** of a stock and classify the outlook as **Bullish / Bearish / Neutral**, using **market index context** (S&P 500 / NIFTY 50).

---

## ✨ What This App Does

For a given stock ticker:

✅ Downloads stock + market index historical data (Yahoo Finance)  
✅ Builds market-aware engineered features  
✅ Trains 2 models:
- **Trend Model (XGBClassifier)** → probability of bullish trend
- **Return Model (XGBRegressor)** → expected cumulative return over next N days  
✅ Saves models locally per ticker  
✅ Predicts:
- bullish probability
- expected N-day return (%)
- rough predicted price
- decision label (Bullish / Bearish / Neutral)

---

## 📂 Project Structure

Return-Based-Stock-Forecaster/
│
├── app.py # Streamlit UI
├── requirements.txt
├── README.md
│
├── src/
│ ├── train_models.py # Train trend + return models
│ ├── predict.py # Predict next N-day return + decision
│ ├── data_loader.py # Fetch + cache yfinance data
│ ├── dataset_market.py # Sequence dataset builder
│ ├── features_market.py # Feature engineering
│ └── market_context.py # Maps ticker → correct market index
│
├── data/ # Cached stock CSV files (auto-generated)
└── models/ # Saved trained models (auto-generated)


---

## ⚙️ Installation (Local Setup)

### 1) Clone the repository
```bash
git clone https://github.com/AdityaV33/Return-Based-Stock-Forecaster.git
cd Return-Based-Stock-Forecaster

python -m venv venv
venv\Scripts\activate

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

streamlit run app.py

---
```

🚀 How To Use
Step 1: Train models

  1.Enter a ticker (example: AAPL or RELIANCE.NS)
  2.Click 🧠 Train Models
  3.Models will be saved inside the models/ folder

Step 2: Predict

  1.Click 📊 Predict
  2.The app loads the saved models and prints the forecast

🧾 Example Output

Ticker: AAPL
Market index: ^GSPC
Horizon (trading days): 10
------------------------------
Trend probability (Bullish): 0.6123
Expected N-day return: 2.418 %
Decision: Bullish
------------------------------
Last Close: 192.55
Predicted Price (rough): 197.20

🌍 Market Index Mapping

The project automatically selects the correct market index:
US / Global stocks → ^GSPC (S&P 500)
NSE India stocks (.NS) → ^NSEI (NIFTY 50)

🧠 Decision Logic

The decision is based on bullish probability:
Bullish 📈 if trend_prob ≥ 0.55
Bearish 📉 if trend_prob ≤ 0.45
Neutral ⚖️ otherwise (no-trade zone)

🧪 Notes & Limitations

⚠️ This project is for educational purposes only (not financial advice).
⚠️ Results can vary depending on volatility and market regime.
⚠️ Yahoo Finance may sometimes rate-limit requests (YFRateLimitError).
⚠️ The predicted price is a rough conversion from predicted return.

🛠️ Tech Stack

Python
Streamlit
XGBoost
scikit-learn
pandas / numpy
yfinance

📌 Future Improvements

Add backtesting & evaluation charts in the UI
Add multi-ticker batch training
Improve threshold calibration across stocks
Improve caching + speed
Deploy on Streamlit Cloud

📜 License

This project is intended for academic / educational use

<img width="1034" height="779" alt="image" src="https://github.com/user-attachments/assets/13ef49ad-29fc-48f4-883b-75d4a0d1bee2" />
<img width="1038" height="813" alt="image" src="https://github.com/user-attachments/assets/52bfc4fd-ec01-4042-95d2-6afa58e457d7" />
<img width="984" height="720" alt="image" src="https://github.com/user-attachments/assets/56b37b87-fe12-4beb-ab02-0d41d809b741" />
<img width="930" height="815" alt="image" src="https://github.com/user-attachments/assets/1dfbbf63-aeaf-42a9-b10e-8eb5c2b00b79" />




