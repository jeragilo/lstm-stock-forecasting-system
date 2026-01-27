import os
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import talib
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def fetch_stock_data(stock_symbol):
    stock_data = yf.download(stock_symbol, period="1y", interval="1d")

    if stock_data.empty:
        raise ValueError(f"❌ No data found for {stock_symbol}")

    stock_data.ffill(inplace=True)
    stock_data.bfill(inplace=True)

    close_prices = stock_data["Close"].values.astype(np.float64).flatten()
    stock_data["RSI"] = talib.RSI(close_prices, timeperiod=14)
    stock_data.fillna(0, inplace=True)

    return stock_data

def train_and_save_models(stock_symbol="AAPL"):
    df = fetch_stock_data(stock_symbol)

    X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI']].values
    y = df[['Close']].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_lstm.pkl"))

    models = {
        "Linear_Regression": LinearRegression(),
        "Random_Forest": RandomForestRegressor(),
        "Gradient_Boosting": GradientBoostingRegressor(),
        "XGBoost": XGBRegressor()
    }

    model_metrics = {}

    for name, model in models.items():
        model.fit(X, y.ravel())
        joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.pkl"))

        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        rmse = sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        model_metrics[name] = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}
        print(f"✅ {name} trained. MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

    # ✅ Train LSTM Model
    class LSTMModel(nn.Module):
        def __init__(self):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size=6, hidden_size=50, num_layers=2, batch_first=True)
            self.fc = nn.Linear(50, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    lstm_model = LSTMModel()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    X_train = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(y, dtype=torch.float32)

    for epoch in range(100):
        optimizer.zero_grad()
        output = lstm_model(X_train)
        loss = loss_function(output, y_train)
        loss.backward()
        optimizer.step()

    torch.save({"lstm_state_dict": lstm_model.state_dict()}, os.path.join(MODEL_DIR, "lstm_model.pth"))

    # ✅ Save LSTM Metrics
    model_metrics["LSTM"] = {"MSE": float(loss.item()), "RMSE": sqrt(float(loss.item())), "MAE": None, "R2": None}
    print(f"✅ LSTM trained. MSE: {loss.item():.4f}, RMSE: {sqrt(loss.item()):.4f}")

    # ✅ Save Metrics
    joblib.dump(model_metrics, os.path.join(MODEL_DIR, "model_metrics.pkl"))
    print("✅ Model Metrics Saved!")

train_and_save_models()
