import os
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import joblib

# ✅ Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# ✅ Fetch Stock Data
def fetch_stock_data(stock_symbol, start_date, end_date):
    df = yf.download(stock_symbol, start=start_date, end=end_date)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    return df

# ✅ Prepare Data
def prepare_data(df, sequence_length=50):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i+sequence_length])
        y.append(scaled_data[i+sequence_length, 3])  # Predicting "Close" price

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), scaler

# ✅ Train Model
def train_lstm(stock_symbol="AAPL"):
    print(f"📈 Training LSTM for {stock_symbol}...")

    # Download data
    df = fetch_stock_data(stock_symbol, start_date="2015-01-01", end_date="2024-01-01")
    
    if df is None or df.empty:
        print("❌ No stock data available!")
        return

    X, y, scaler = prepare_data(df)

    # Define model parameters
    input_size = X.shape[2]
    hidden_size = 50
    num_layers = 2
    output_size = 1
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train model
    epochs = 20
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions.squeeze(), y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    # ✅ Save model and scaler
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(models_dir, "lstm_model.pth"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler_lstm.pkl"))
    print(f"✅ LSTM Model saved to {models_dir}/lstm_model.pth")

# ✅ Run training
if __name__ == "__main__":
    train_lstm()

