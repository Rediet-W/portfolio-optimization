import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

def calculate_risk_metrics(returns):
    sharpe = (returns.mean() / returns.std()) * (252**0.5)
    var_95 = returns.quantile(0.05)
    return pd.DataFrame({'Sharpe Ratio': sharpe, '95% VaR': var_95})

def test_stationarity(series):
    result = adfuller(series.dropna())
    return {"ADF Statistic": result[0], "p-value": result[1]}