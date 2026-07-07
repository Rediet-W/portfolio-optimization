import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt import risk_models, expected_returns, EfficientFrontier, plotting

# 1. Load and Prepare Data
df = pd.read_csv('data/processed/market_data.csv', header=[0, 1], index_col=0, parse_dates=True)
prices = df.xs('Close', level='Price', axis=1)[['TSLA', 'SPY', 'BND']]

# 2. Setup Parameters
# Convert price to annualized return rate: (Forecast - Current) / Current
tsla_current = prices['TSLA'].iloc[-1]
tsla_annual_ret = (300.0 - tsla_current) / tsla_current
mu = expected_returns.mean_historical_return(prices)
mu['TSLA'] = tsla_annual_ret 
S = risk_models.sample_cov(prices)

# 3. Heatmap Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(S, annot=True, cmap='coolwarm')
plt.title("Covariance Matrix Heatmap")
plt.show()

# 4. Separate Optimizations
# Max Sharpe Ratio
ef_max = EfficientFrontier(mu, S)
weights_max = ef_max.max_sharpe()
print("Max Sharpe Weights:", ef_max.clean_weights())
ef_max.portfolio_performance(verbose=True)

# Min Volatility
ef_min = EfficientFrontier(mu, S)
weights_min = ef_min.min_volatility()
print("Min Volatility Weights:", ef_min.clean_weights())
ef_min.portfolio_performance(verbose=True)

# 5. Plotting the Efficient Frontier
# 5. Plotting the Efficient Frontier
fig, ax = plt.subplots(figsize=(10, 6))

# Use a CLEAN, UNRESOLVED instance for the plotting function
ef_plot = EfficientFrontier(mu, S) 
plotting.plot_efficient_frontier(ef_plot, ax=ax, show_assets=True)

# Overlay the specific portfolios using your already-solved instances
ax.scatter(ef_min.portfolio_performance()[1], ef_min.portfolio_performance()[0], marker="o", s=100, c="g", label="Min Volatility")
ax.scatter(ef_max.portfolio_performance()[1], ef_max.portfolio_performance()[0], marker="*", s=150, c="r", label="Max Sharpe")

plt.title("Efficient Frontier")
plt.legend()
plt.show()