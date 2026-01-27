import yfinance as yf
import talib
import pandas as pd
import numpy as np

def fetch_stock_data(stock_symbol):
    """Fetch stock data and compute technical indicators."""
    stock_data = yf.download(stock_symbol, period="10y", interval="1d")

    # ✅ Ensure data is available
    if stock_data.empty:
        raise ValueError(f"❌ No data found for {stock_symbol}")

    stock_data.ffill(inplace=True)
    stock_data.bfill(inplace=True)

    close_prices = stock_data["Close"].astype(float).values.flatten()
    
    # ✅ Compute Technical Indicators
    stock_data["RSI"] = talib.RSI(close_prices, timeperiod=14)
    stock_data["SMA_5"] = talib.SMA(close_prices, timeperiod=5)
    stock_data["EMA_5"] = talib.EMA(close_prices, timeperiod=5)
    
    # ✅ Compute Volatility and ensure it's a single column
    stock_data["Volatility"] = stock_data["Close"].rolling(10).std()
    stock_data["Volatility"] = stock_data["Volatility"].astype(float)  # ✅ Ensure it's a Series

    # ✅ Compute Momentum
    stock_data["Momentum"] = stock_data["Close"].diff(5)

    # ✅ Compute MACD
    macd, macd_signal, _ = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
    stock_data["MACD"] = macd - macd_signal  

    # ✅ **Fix Implied Volatility Calculation (Force Single Column)**
    stock_data["Implied_Volatility"] = stock_data["Volatility"].squeeze().astype(float) / (stock_data["Close"].squeeze().astype(float) + 1e-6)

    stock_data.fillna(0, inplace=True)

    # ✅ **Flatten MultiIndex Columns**
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in stock_data.columns.values]

    # ✅ Debug Output
    print("\n📊 Features in Dataset (Fixed):")
    print(stock_data.head())
    print("\n🔍 Feature Columns:", list(stock_data.columns))

    return stock_data

# ✅ Run Debug Test Again
fetch_stock_data("AAPL")
