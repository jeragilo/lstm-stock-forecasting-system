# LSTM Stock Forecasting System

A full-stack machine-learning project for experimenting with financial time-series forecasting using **PyTorch LSTMs**, technical indicators, **ONNX inference**, a **Flask API**, and a **React frontend**.

This repository is intended as an applied ML/software-engineering project. It demonstrates the path from data acquisition and feature engineering through model training, inference, API integration, and user-facing presentation.

> **Scope:** This is an experimental forecasting system and portfolio project, not financial advice or a production trading system.

## What the Project Demonstrates

- Historical market-data acquisition with `yfinance`
- Financial feature engineering and technical indicators
- LSTM model training with PyTorch
- Model export and inference with ONNX Runtime
- Prediction/inference validation scripts
- Flask-based backend integration
- React-based frontend interface
- Separation of training, inference, application, and debugging workflows

## System Architecture

```text
Historical market data
        |
        v
Feature engineering / preprocessing
        |
        v
PyTorch LSTM training
        |
        +------------------+
        |                  |
        v                  v
PyTorch model         ONNX export
                           |
                           v
                     ONNX inference
                           |
                           v
                       Flask API
                           |
                           v
                     React frontend
```

## Technology Stack

| Layer | Technologies |
|---|---|
| Machine learning | Python, PyTorch, NumPy, Pandas |
| Market data / features | yfinance, TA-Lib |
| Inference | ONNX, ONNX Runtime |
| Backend | Flask |
| Frontend | React, JavaScript, HTML, CSS |
| Development | Git, Conda, Joblib |

## Financial Features

The project explores technical and statistical features including:

- Relative Strength Index (RSI)
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Momentum
- Volatility
- Moving Average Convergence Divergence (MACD)

The repository also contains utilities for downloading historical stock data, handling missing observations, and preparing features for downstream model workflows.

## Repository Structure

```text
lstm-stock-forecasting-system/
├── backend/
│   ├── app.py
│   ├── benchmark_models.py
│   ├── debug_fetch_stock.py
│   ├── requirements.txt
│   ├── test_lstm_predictions.py
│   ├── test_onnx_inference.py
│   ├── trade_model.py
│   ├── trade_models.py
│   ├── train_lstm.py
│   └── train_models.py
├── stock-price-frontend-manual/
└── README.md
```

## Quick Start

Clone the repository:

```bash
git clone https://github.com/jeragilo/lstm-stock-forecasting-system.git
cd lstm-stock-forecasting-system
```

Install the backend dependencies:

```bash
cd backend
pip install -r requirements.txt
```

The repository contains multiple experimental training and inference scripts. Two useful entry points are:

```bash
python train_lstm.py
python test_onnx_inference.py
```

To run the frontend, install its JavaScript dependencies from the frontend directory:

```bash
cd ../stock-price-frontend-manual
npm install
npm start
```

## Engineering Emphasis

The value of this project is not a claim that an LSTM can reliably predict financial markets. Instead, the repository demonstrates an end-to-end ML engineering workflow:

1. acquire real time-series data;
2. construct numerical features;
3. train a recurrent model;
4. export the model into a portable inference format;
5. validate inference behavior;
6. expose model functionality through an API; and
7. connect that API to a frontend application.

This makes the project complementary to my research-oriented quantum-computing work and my high-performance-computing projects.

## Current Limitations

- Financial markets are noisy, non-stationary systems; historical predictive performance does not imply future performance.
- The repository is an experimental portfolio system rather than a production trading platform.
- Model evaluation should be expanded with stronger naive/statistical baselines, walk-forward validation, and explicit out-of-sample metrics before drawing forecasting conclusions.
- Deployment hardening, automated tests, CI, monitoring, and model-version management would be required for production use.

## Author

**Jesús Gil**  
Computer Science · Applied Mathematics · Quantum Computing · Machine Learning · HPC

[GitHub](https://github.com/jeragilo) · [LinkedIn](https://www.linkedin.com/in/jesusrgil)
