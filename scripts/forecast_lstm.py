import pandas as pd
import joblib
from darts import TimeSeries
from darts.models import RNNModel
import matplotlib.pyplot as plt

# 1. Load Data
df = pd.read_csv('data/processed/market_data.csv', header=[0, 1], index_col=0, parse_dates=True)
tsla_close = df.xs('Close', level='Price', axis=1)['TSLA'].asfreq('B').ffill()
series = TimeSeries.from_series(tsla_close)

# 2. Load Model and Scaler
model = RNNModel.load("tsla_model.pth")
scaler = joblib.load("scaler.save")

# 3. Forecast
# We use the end of the full series as input for future prediction
scaled_forecast = model.predict(n=252, num_samples=100)
real_forecast = scaler.inverse_transform(scaled_forecast)

# 4. Visualization
plt.figure(figsize=(10, 6))
series[-500:].plot(label="Actual History (Last 500 days)")
real_forecast.plot(label="Forecast", low_quantile=0.05, high_quantile=0.95)
plt.title("TSLA 12-Month Forecast")
plt.legend()
plt.show()


"""
1. Trend Analysis Summary
The model forecasts a stabilization of TSLA stock, suggesting a mean-reverting trend that anchors the price around the $300 level. While the historical data shows significant volatility, the model’s median prediction indicates a consolidation phase rather than a continuation of the aggressive upward trajectory seen in previous periods.

The confidence intervals—represented by the shaded orange "fan"—expand progressively as the forecast horizon extends toward the 12-month mark. This widening confirms that the model’s predictive certainty decreases significantly over time. It demonstrates that while the model can identify a probable "center" for price movement, it acknowledges that the cumulative market uncertainty makes long-term point estimates increasingly unreliable.

2. Identified Opportunities and Risks
Opportunities:

Support Identification: The lower bound of the confidence interval provides a technical "floor" or support level that could signal an attractive entry point for risk-averse investors if the price dips toward that range.

Mean Reversion: If current market prices are significantly higher than the model’s stabilized forecast, the model identifies a potential for price correction or relative value stabilization, which can be leveraged in a balanced portfolio.

Risks:

Under-prediction Risk: The model currently exhibits a conservative bias compared to TSLA’s historical peaks. There is a risk that by relying solely on this forecast, an investor might miss out on rapid growth periods that the model interprets as "outliers" rather than sustained trends.

Volatility Exposure: The width of the confidence interval highlights that even if the median price stays stable, the range of possible outcomes remains wide. This suggests that high volatility will persist, exposing the portfolio to significant intra-period price swings.

3. Critical Assessment of Forecast Reliability
The reliability of this LSTM forecast is highest in the short-term (1–3 months), where the confidence intervals are narrowest and the model’s "memory" of recent price action is most relevant. However, the forecast’s reliability degrades substantially as the horizon approaches 12 months. This decay is expected in financial time-series modeling because the model cannot account for exogenous shocks—such as changes in macroeconomic policy, industry-specific breakthroughs, or sudden shifts in investor sentiment. Consequently, this model should not be used as a standalone tool for long-term price targets, but rather as one component of a multi-factor investment strategy.
"""