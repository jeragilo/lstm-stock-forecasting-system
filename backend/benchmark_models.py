import yfinance as yf
import talib
import numpy as np
import pandas as pd

def fetch_stock_data(stock_symbol):
    """Download stock data and compute features."""
    print(f"📥 Fetching data for {stock_symbol}...")
    stock_data = yf.download(stock_symbol, period="10y", interval="1d")

    if stock_data.empty:
        raise ValueError(f"❌ No data found for {stock_symbol}")

    # ✅ Fill missing values before calculations
    stock_data.ffill(inplace=True)
    stock_data.bfill(inplace=True)

    # ✅ Ensure Close prices are a 1D NumPy array before passing to talib.RSI()
    close_prices = stock_data["Close"].astype(float).values.flatten()

    # ✅ Compute RSI properly (fixing 0 values)
    stock_data["RSI"] = talib.RSI(close_prices, timeperiod=14)
    
    # ✅ Drop initial NaN RSI values (first 14 days)
    stock_data.dropna(inplace=True)

    print(f"✅ Successfully computed RSI for {stock_symbol}.")
    return stock_data

# ✅ Test Execution
if __name__ == "__main__":
    stock_symbol = "AAPL"
    df = fetch_stock_data(stock_symbol)
    print("📊 Sample Data:\n", df.head())
