import os
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import talib
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

# ✅ Directory Setup
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def fetch_stock_data(stock_symbol):
    """Download stock data and compute indicators."""
    stock_data = yf.download(stock_symbol, period="10y", interval="1d")
    if stock_data.empty:
        raise ValueError(f"❌ No data found for {stock_symbol}")
    stock_data.ffill(inplace=True)
    stock_data.bfill(inplace=True)
    stock_data = stock_data.infer_objects(copy=False)
    stock_data["RSI"] = talib.RSI(stock_data["Close"].values, timeperiod=14)
    stock_data.fillna(0, inplace=True)
    return stock_data

def train_and_save_models(stock_symbol="AAPL"):
    """Train models and save them with metrics."""
    df = fetch_stock_data(stock_symbol)
    X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI']].values
    y = df[['Close']].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_lstm.pkl"))
    models = {
        "Linear_Regression": LinearRegression(),
        "Random_Forest": RandomForestRegressor(),
        "Gradient_Boosting": GradientBoostingRegressor(),
        "XGBoost": XGBRegressor()
    }
    model_metrics = {}
    for name, model in models.items():
        model.fit(X_train, y_train.ravel())
        joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.pkl"))
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        model_metrics[name] = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}
        print(f"✅ {name} trained. MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class LSTMModel(nn.Module):
        def __init__(self):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size=6, hidden_size=50, num_layers=2, batch_first=True)
            self.fc = nn.Linear(50, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])
    lstm_model = LSTMModel().to(device)
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
    loss_function = nn.MSELoss()
    X_train_torch = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)
    y_train_torch = torch.tensor(y_train, dtype=torch.float32).to(device)
    for _ in range(100):
        optimizer.zero_grad()
        output = lstm_model(X_train_torch)
        loss = loss_function(output, y_train_torch)
        loss.backward()
        optimizer.step()
    torch.save({"lstm_state_dict": lstm_model.state_dict()}, os.path.join(MODEL_DIR, "lstm_model.pth"))
    print("✅ LSTM Model Saved!")
    with torch.no_grad():
        X_test_torch = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
        y_test_torch = torch.tensor(y_test, dtype=torch.float32).to(device)
        y_pred_lstm = lstm_model(X_test_torch).cpu().numpy()
        mse_lstm = mean_squared_error(y_test, y_pred_lstm)
        rmse_lstm = sqrt(mse_lstm)
        mae_lstm = mean_absolute_error(y_test, y_pred_lstm)
        r2_lstm = r2_score(y_test, y_pred_lstm)
        model_metrics["LSTM"] = {"MSE": mse_lstm, "RMSE": rmse_lstm, "MAE": mae_lstm, "R2": r2_lstm}
        print(f"✅ LSTM trained. MSE: {mse_lstm:.4f}, RMSE: {rmse_lstm:.4f}, MAE: {mae_lstm:.4f}, R²: {r2_lstm:.4f}")
    joblib.dump(model_metrics, os.path.join(MODEL_DIR, "model_metrics.pkl"))
    print("✅ All Model Metrics Saved!")
train_and_save_models()
