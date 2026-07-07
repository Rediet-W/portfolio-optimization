import pandas as pd
import matplotlib.pyplot as plt

# 1. Load data for the last year (e.g., 2025-06 to 2026-06)
df = pd.read_csv('data/processed/market_data.csv', header=[0, 1], index_col=0, parse_dates=True)
returns = df.xs('Close', level='Price', axis=1)[['TSLA', 'SPY', 'BND']].pct_change().dropna()
test_returns = returns.loc['2025-06-01':'2026-06-01']

# 2. Define Weights
# Your Optimal Weights from Task 4
strategy_weights = pd.Series({'TSLA': 0.0, 'SPY': 0.45, 'BND': 0.55}) 
# Benchmark: 60% SPY / 40% BND
benchmark_weights = pd.Series({'TSLA': 0.0, 'SPY': 0.60, 'BND': 0.40})

# 3. Calculate Portfolio Returns
strategy_ret = (test_returns * strategy_weights).sum(axis=1)
benchmark_ret = (test_returns * benchmark_weights).sum(axis=1)

# 4. Cumulative Returns
cumulative_strategy = (1 + strategy_ret).cumprod()
cumulative_benchmark = (1 + benchmark_ret).cumprod()

# 5. Plot
cumulative_strategy.plot(label='Optimized Strategy')
cumulative_benchmark.plot(label='60/40 Benchmark')
plt.legend()
plt.show()