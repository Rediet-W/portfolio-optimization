import matplotlib.pyplot as plt
from pypfopt import plotting, EfficientFrontier

# 1. Setup the Efficient Frontier object
# Note: Ensure ef is initialized with the same mu and S as your optimizer
ef = EfficientFrontier(mu, S) 

# 2. Plot the Efficient Frontier
fig, ax = plt.subplots(figsize=(10, 6))
plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)

# 3. Find and mark key portfolios
# Max Sharpe Ratio
ef.max_sharpe()
ret_tangent, std_tangent, _ = ef.portfolio_performance()
ax.scatter(std_tangent, ret_tangent, marker="*", s=150, c="r", label="Max Sharpe")

# Min Volatility
ef.min_volatility()
ret_minvol, std_minvol, _ = ef.portfolio_performance()
ax.scatter(std_minvol, ret_minvol, marker="*", s=150, c="g", label="Min Volatility")

plt.title("Efficient Frontier with Optimal Portfolios")
plt.legend()
plt.show()