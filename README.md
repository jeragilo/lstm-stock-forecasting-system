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

## 📁 Directory Structure

stock-price-predictor/
│
├── backend/
│ ├── app.py # Flask API
│ ├── train_models.py # LSTM model training
│ ├── benchmark_models.py # Model evaluation (MSE)
│ ├── test_lstm_predictions.py # ONNX model inference
│ ├── trade_models.py # Extended prediction logic
│ ├── models/ # Saved .pth and scaler.pkl
│ ├── .gitignore
│ └── requirements.txt
│
├── stock-price-frontend-manual/ (or frontend/)
│ ├── App.js
│ ├── StockForm.js
│ ├── index.js
│ ├── index.html
│ ├── package.json
│ └── package-lock.json
│
└── README.md


---

## 📈 Demo: How It Works

1. Clone the repo:
   ```bash
   git clone git@github.com:jeragilo/stock-price-predictor.git
   cd stock-price-predictor
Train the model (LSTM on AAPL):

cd backend
python train_models.py
Predict closing price:

python test_lstm_predictions.py
Run the frontend:

cd ../stock-price-frontend-manual
npm install
npm start
