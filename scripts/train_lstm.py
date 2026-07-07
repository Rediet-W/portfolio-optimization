import pandas as pd
from darts import TimeSeries
from darts.models import RNNModel
from darts.utils.likelihood_models import GaussianLikelihood
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mae, rmse, mape
import joblib

# 1. Load and Prepare Data
df = pd.read_csv('data/processed/market_data.csv', header=[0, 1], index_col=0, parse_dates=True)
tsla_close = df.xs('Close', level='Price', axis=1)['TSLA'].asfreq('B').ffill()
series = TimeSeries.from_series(tsla_close)

# 2. Split Data
train, test = series.split_before(0.8)

# 3. Scale Data (Fit on Train ONLY)
scaler = Scaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

# 4. Train LSTM
model = RNNModel(
    model='LSTM',
    hidden_dim=50,
    n_epochs=20,
    batch_size=32,
    input_chunk_length=60,
    output_chunk_length=1,
    training_length=60,
    likelihood=GaussianLikelihood(),
    optimizer_kwargs={'lr': 1e-3},
    random_state=42
)

model.fit(train_scaled)

# 5. Save Model and Scaler
model.save("tsla_model.pth")
joblib.dump(scaler, "scaler.save")
print("Model and Scaler saved successfully.")

# 6. Evaluate
forecast_scaled = model.predict(len(test))
forecast = scaler.inverse_transform(forecast_scaled)

print(f"MAE: {mae(test, forecast)}")
print(f"RMSE: {rmse(test, forecast)}")
print(f"MAPE: {mape(test, forecast)}")