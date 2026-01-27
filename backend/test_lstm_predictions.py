import torch
import numpy as np
import pandas as pd
import yfinance as yf
import pickle
import onnxruntime as ort
from train_models import LSTMModel, FEATURE_COLUMNS  # ✅ FIXED IMPORT

# ✅ Load Scaler
with open("models/scaler_lstm.pkl", "rb") as f:
    scaler = pickle.load(f)

# ✅ Fetch Latest Stock Data
def fetch_latest_data(stock_symbol):
    stock_data = yf.download(stock_symbol, period="10d", interval="1d")

    if stock_data.empty:
        raise ValueError(f"❌ No data found for {stock_symbol}")

    stock_data.ffill(inplace=True)
    stock_data.bfill(inplace=True)

    features = stock_data[FEATURE_COLUMNS].values  # ✅ FIXED: Using Correct Features
    features_scaled = scaler.transform(features).astype(np.float32)
    
    return features_scaled[-1].reshape(1, 1, -1)  # ✅ Ensuring Correct ONNX Input Shape

# ✅ Load ONNX Model
session = ort.InferenceSession("models/lstm_model.onnx", providers=["CPUExecutionProvider"])

# ✅ Predict Stock Price
input_data = fetch_latest_data("AAPL")
output = session.run(None, {"input": input_data})[0]

# ✅ Reverse Scaling to Get Real Price
predicted_price = scaler.inverse_transform(np.zeros((1, len(FEATURE_COLUMNS))))[0, 3] + output[0, 0]

print(f"📈 Final Predicted Closing Price: ${predicted_price:.2f}")
