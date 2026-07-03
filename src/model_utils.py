import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def split_data(series, train_size=0.8):
    split = int(len(series) * train_size)
    return series[:split], series[split:]

def evaluate_forecast(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}