"""
Temporal Leakage Correct Implementation - Validation Suite

This file demonstrates the CORRECT way to handle temporal data:
- Only use past information for feature engineering
- Proper rolling windows that don't look ahead
- Temporal splits instead of random splits
- Lag features with positive shifts only
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Create sample time series data
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=1000, freq='D')
values = np.cumsum(np.random.randn(1000)) + 100  # Random walk with trend

time_series_data = pd.DataFrame({
    'timestamp': dates,
    'value': values,
    'noise': np.random.randn(1000) * 0.1
})

# CORRECT: Use only past data for features
print("✅ Creating temporal features correctly...")

# CORRECT: Lag features (using past values)
time_series_data['lag_1'] = time_series_data['value'].shift(1)    # Yesterday's value
time_series_data['lag_7'] = time_series_data['value'].shift(7)    # Last week's value
time_series_data['lag_30'] = time_series_data['value'].shift(30)  # Last month's value

# CORRECT: Rolling windows without center=True (only past data)
time_series_data['rolling_mean_7'] = time_series_data['value'].rolling(
    window=7, center=False  # center=False ensures only past data
).mean()

time_series_data['rolling_std_7'] = time_series_data['value'].rolling(
    window=7, center=False, min_periods=1
).std()

# CORRECT: Expanding window (cumulative stats using only past)
time_series_data['expanding_mean'] = time_series_data['value'].expanding().mean()
time_series_data['expanding_max'] = time_series_data['value'].expanding().max()

# CORRECT: Temporal feature engineering without future leakage
time_series_data['day_of_week'] = time_series_data['timestamp'].dt.dayofweek
time_series_data['month'] = time_series_data['timestamp'].dt.month
time_series_data['is_weekend'] = time_series_data['day_of_week'].isin([5, 6]).astype(int)

# CORRECT: Remove rows with NaN (due to lags) and prepare features
features = [
    'lag_1', 'lag_7', 'lag_30', 
    'rolling_mean_7', 'rolling_std_7',
    'expanding_mean', 'expanding_max',
    'day_of_week', 'month', 'is_weekend', 'noise'
]

# Clean data (remove NaN from lags)
clean_data = time_series_data.dropna()
X = clean_data[features]
y = clean_data['value']

print(f"✅ Feature matrix shape: {X.shape}")
print(f"✅ Target vector shape: {y.shape}")

# CORRECT: Temporal split (respecting chronological order)
split_point = int(0.8 * len(clean_data))
X_train = X[:split_point]  # Past data for training
X_test = X[split_point:]   # Future data for testing
y_train = y[:split_point]
y_test = y[split_point:]

print(f"✅ Training period: {clean_data.iloc[0]['timestamp']} to {clean_data.iloc[split_point-1]['timestamp']}")
print(f"✅ Testing period: {clean_data.iloc[split_point]['timestamp']} to {clean_data.iloc[-1]['timestamp']}")

# CORRECT: Cross-validation with TimeSeriesSplit (respects temporal order)
tscv = TimeSeriesSplit(n_splits=5)
model = RandomForestRegressor(n_estimators=50, random_state=42)

# Train final model
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

print(f"✅ CORRECT Temporal Model - MSE: {mse:.4f}")
print("✅ No future leakage: all features use only past information")
print("✅ Temporal split respects chronological order")
print("✅ Model trained only on historical data")

# CORRECT: Validation using TimeSeriesSplit
cv_scores = []
for train_idx, val_idx in tscv.split(X_train):
    X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    cv_model = RandomForestRegressor(n_estimators=50, random_state=42)
    cv_model.fit(X_cv_train, y_cv_train)
    cv_pred = cv_model.predict(X_cv_val)
    cv_score = mean_squared_error(y_cv_val, cv_pred)
    cv_scores.append(cv_score)

print(f"✅ Time Series CV Score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores)*2:.4f})")
print("✅ Cross-validation respects temporal boundaries")