import sys
import os

# Add the parent directory (the root of your project) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from pmdarima import auto_arima
from src.model_utils import split_data, evaluate_forecast

# Load data
df = pd.read_csv('data/processed/market_data.csv', header=[0, 1], index_col=0, parse_dates=True)
tsla_close = df.xs('Close', level='Price', axis=1)['TSLA']

tsla_close = tsla_close.asfreq('B').ffill()

train, test = split_data(tsla_close)

# Auto ARIMA
model = auto_arima(train, seasonal=False, stepwise=True, suppress_warnings=True)
print(model.summary())

# Forecast
forecast = model.predict(n_periods=len(test))
metrics = evaluate_forecast(test.values, forecast)
print("ARIMA Metrics:", metrics)