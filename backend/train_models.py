import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import yfinance as yf
import talib
import pickle
import os
import onnx
import onnxruntime as ort
from sklearn.preprocessing import StandardScaler

# ✅ Hyperparameters
input_size = 12
hidden_size = 512
num_layers = 3
learning_rate = 0.0005
batch_size = 32
num_epochs = 2000

# ✅ Feature Columns (Explicitly Defined)
FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Volume", "RSI", "SMA_5", "EMA_5", "Volatility", "Momentum", "MACD", "Implied_Volatility"]

# ✅ LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out.squeeze()

# ✅ Fetch Stock Data (Fix Missing Columns)
def fetch_stock_data(stock_symbol, retries=3):
    for attempt in range(retries):
        print(f"📥 Attempting to fetch {stock_symbol} data... (Try {attempt+1}/{retries})")
        stock_data = yf.download(stock_symbol, period="10y", interval="1d")

        if stock_data.empty:
            print(f"❌ No data returned for {stock_symbol}. Retrying...")
            continue  # Retry fetching

        stock_data = stock_data.reset_index()
        stock_data.columns = stock_data.columns.map(lambda x: x[0] if isinstance(x, tuple) else x)

        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        missing_columns = [col for col in required_columns if col not in stock_data.columns]

        if missing_columns:
            print(f"⚠️ WARNING: Missing columns {missing_columns}. Retrying...")
            continue  # Retry fetching

        close_prices = stock_data["Close"].astype(float).values.flatten()

        # ✅ Compute Features
        stock_data["RSI"] = talib.RSI(close_prices, timeperiod=14)
        stock_data["SMA_5"] = talib.SMA(close_prices, timeperiod=5)
        stock_data["EMA_5"] = talib.EMA(close_prices, timeperiod=5)
        stock_data["Volatility"] = stock_data["Close"].rolling(10).std()
        stock_data["Momentum"] = stock_data["Close"].diff(5)
        macd, macd_signal, _ = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
        stock_data["MACD"] = macd - macd_signal
        stock_data["Implied_Volatility"] = stock_data["Volatility"].fillna(0) / (stock_data["Close"].fillna(1) + 1e-6)

        stock_data.fillna(0, inplace=True)
        return stock_data

    raise ValueError(f"❌ Failed to fetch valid stock data after {retries} attempts.")

# ✅ Train & Save Model
def train_and_save_models():
    print("📥 Fetching stock data for training...")
    df = fetch_stock_data("AAPL")

    # ✅ Ensure feature consistency
    feature_columns = [col for col in FEATURE_COLUMNS if col in df.columns]
    assert len(feature_columns) == 12, f"❌ Feature mismatch! Expected 12 but got {len(feature_columns)}. Features found: {feature_columns}"

    features = df[feature_columns].values
    labels = df["Close"].values

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    os.makedirs("models", exist_ok=True)
    with open("models/scaler_lstm.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # ✅ Convert to Tensors
    X_train = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(1).to(device)
    y_train = torch.tensor(labels, dtype=torch.float32).to(device)

    # ✅ Create Model
    model = LSTMModel(input_size, hidden_size, num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ✅ Training Loop
    best_loss = float("inf")
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # ✅ Early Stopping Logic
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), "models/best_lstm_model.pth")

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # ✅ Save Model
    torch.save(model.state_dict(), "models/lstm_model.pth")
    print("✅ LSTM Model Saved!")

    # ✅ Convert to ONNX
    dummy_input = torch.randn(1, 1, input_size).to(device)
    onnx_path = "models/lstm_model.onnx"
    torch.onnx.export(model, dummy_input, onnx_path, input_names=["input"], output_names=["output"])
    print(f"✅ LSTM Model Converted to ONNX at {onnx_path}")

# ✅ Run Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_and_save_models()
