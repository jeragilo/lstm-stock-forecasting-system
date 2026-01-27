![GitHub last commit](https://img.shields.io/github/last-commit/jeragilo/stock-price-predictor)
![GitHub Repo stars](https://img.shields.io/github/stars/jeragilo/stock-price-predictor?style=social)
![GitHub forks](https://img.shields.io/github/forks/jeragilo/stock-price-predictor?style=social)
![GitHub issues](https://img.shields.io/github/issues/jeragilo/stock-price-predictor)

# 📈 Stock Price Predictor: LSTM & ONNX-Powered ML Forecasting App

A full-stack machine learning project for predicting stock prices using LSTM neural networks. Built with a PyTorch backend and a React.js frontend, this project showcases advanced time series modeling, ONNX conversion, and seamless user interaction.

---

## 🚀 Key Features

- 🔁 Long Short-Term Memory (LSTM) model trained on historical stock data
- 🧠 Deep learning with PyTorch and ONNX export
- 🧪 Model testing, benchmarking, and ONNX inference validation
- 🖥️ Full React.js frontend with stock input form and prediction results
- 📉 Integrated technical indicators: RSI, SMA, EMA, Momentum, Volatility, MACD
- 🔍 Scalable architecture for additional tickers and financial metrics

---

## 🛠️ Tech Stack

| Component   | Stack                                          |
|-------------|------------------------------------------------|
| **Frontend**| React.js, JavaScript, HTML, CSS                |
| **Backend** | Python, PyTorch, ONNX, yfinance, TA-Lib, Flask |
| **ML/AI**   | LSTM (Time Series Forecasting), ONNX Runtime   |
| **Dev Tools** | Git, GitHub, VS Code, SSH, Conda, Joblib     |

---

<pre>
## 📁 Directory Structure

<code>
stock-price-predictor/
├── backend/
│   ├── app.py                  # Flask API endpoint
│   ├── train_models.py         # LSTM model training script
│   ├── test_lstm_predictions.py# ONNX inference
│   ├── benchmark_models.py     # Evaluate model performance (MSE)
│   ├── trade_models.py         # Strategy logic & future prediction
│   ├── models/                 # Saved PyTorch & ONNX models
│   ├── requirements.txt        # Python dependencies
│   └── .gitignore
│
├── stock-price-frontend-manual/
│   ├── App.js
│   ├── StockForm.js
│   ├── index.js
│   ├── index.html
│   ├── package.json
│   └── package-lock.json
│
└── README.md
</code>
</pre>



---

## 📈 Demo: How It Works

1. Clone the repo:
   ```bash
   git clone git@github.com:jeragilo/stock-price-predictor.git
   cd stock-price-predictor
<pre>
## 🧪 Demo: How It Works

1. **Clone the Repository**

```bash
git clone git@github.com:jeragilo/stock-price-predictor.git
cd stock-price-predictor
```

2. **Train the LSTM Model (on AAPL)**

```bash
cd backend
python train_models.py
```

3. **Test Predictions (ONNX model)**

```bash
python test_lstm_predictions.py
```

4. **Launch the Frontend**

```bash
cd ../stock-price-frontend-manual
npm install
npm start
```
</pre>

