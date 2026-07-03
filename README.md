### Project: Tesla Stock Forecasting & Portfolio Optimization

This project implements a quantitative finance pipeline to analyze, forecast, and optimize a stock portfolio containing TSLA, SPY, and BND.

# Task 1: Data Acquisition & Preprocessing

Objective: Extract market data and ensure stationarity for modeling.

Key Steps: * Fetched historical data using yfinance.Performed stationarity checks (ADF Test).Handled missing data via asfreq('B') and ffill().Deliverable: data/processed/market_data.csv

# Task 2: Time Series ForecastingObjective: Predict future stock prices using classical and deep learning models.

Models Implemented:ARIMA (SARIMAX 5,1,2): Used as the baseline statistical model.LSTM (RNNModel): A neural network approach to capture non-linear volatility.

Model Comparison Table:
MetricARIMA (SARIMAX 5,1,2)LSTM (RNNModel)MAE149.06254.80RMSE175.07271.71MAPE40.38%75.96%Model Selection RationaleThe ARIMA model outperformed the LSTM in this specific implementation. Financial time series, specifically daily prices, are heavily influenced by the "Random Walk" theory. The SARIMAX model successfully utilized the integrated component ($d=1$) to track price trends, while the LSTM model—despite its high complexity—was prone to overfitting on the noise inherent in daily stock volatility.Repository Structure/data/processed/: Cleaned dataset./scripts/: Python implementation scripts./src/: Utility functions (model_utils.py)./notebooks/: Exploratory Data Analysis.How to RunInstall Requirements: pip install -r requirements.txtRun ARIMA: python scripts/train_arima.pyRun LSTM: python scripts/train_lstm.py