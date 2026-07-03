import sys
import os
import pandas as pd
from darts import TimeSeries
from darts.models import RNNModel

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# 1. Load Data
df = pd.read_csv('data/processed/market_data.csv', header=[0, 1], index_col=0, parse_dates=True)
tsla_close = df.xs('Close', level='Price', axis=1)['TSLA']

# Debug: Print rows after load
print(f"Rows after load: {len(tsla_close)}")

# 2. Resample and Fill
tsla_close.index = pd.to_datetime(tsla_close.index)
tsla_close = tsla_close.sort_index()
tsla_close = tsla_close.asfreq('B').ffill()

# Debug: Print rows after freq adjustment
print(f"Rows after asfreq('B'): {len(tsla_close)}")

# 3. Create TimeSeries
series = TimeSeries.from_series(tsla_close)

# 4. Split
train, test = series.split_before(0.8)

# Debug: Check the split result
print(f"Training set length: {len(train)}") 
print(f"Testing set length: {len(test)}")

# 4. Train LSTM (Darts calls it RNNModel with model='LSTM')
# Darts handles the scaling and windowing internally
# 4. Train LSTM
model = RNNModel(
    model='LSTM',
    hidden_dim=50,
    n_epochs=20,
    batch_size=32,
    input_chunk_length=60,
    output_chunk_length=1,
    training_length=100,  # Explicitly set this to a value > input_chunk_length
    optimizer_kwargs={'lr': 1e-3},
    random_state=42
)

model.fit(train)

# 5. Forecast and Evaluate
forecast = model.predict(len(test))

# Darts makes evaluation easy with built-in metrics
from darts.metrics import mae, rmse, mape
print(f"MAE: {mae(test, forecast)}")
print(f"RMSE: {rmse(test, forecast)}")
print(f"MAPE: {mape(test, forecast)}")

