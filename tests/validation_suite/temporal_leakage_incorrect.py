"""
Temporal Leakage Incorrect Implementation - Validation Suite

This file demonstrates INCORRECT temporal data handling that should be detected:
- Using future data for feature engineering (look-ahead bias)
- Rolling windows with center=True (uses future data)
- Negative shifts (future values)
- Random splits on temporal data
- Global statistics across entire time range
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  # WRONG: should use temporal split
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

print("❌ Creating temporal features with FUTURE LEAKAGE...")

# INCORRECT: Using future data with negative shifts
time_series_data['future_value_1'] = time_series_data['value'].shift(-1)   # FUTURE LEAKAGE!
time_series_data['future_value_7'] = time_series_data['value'].shift(-7)   # FUTURE LEAKAGE!
time_series_data['future_value_30'] = time_series_data['value'].shift(-30) # FUTURE LEAKAGE!

# INCORRECT: Rolling windows with center=True (uses future data)
time_series_data['rolling_centered_7'] = time_series_data['value'].rolling(
    window=7, center=True  # BUG: center=True uses future data!
).mean()

time_series_data['rolling_centered_15'] = time_series_data['value'].rolling(
    window=15, center=True  # BUG: Uses 7 days future + 7 days past
).std()

# INCORRECT: Global statistics computed on entire dataset
global_mean = time_series_data['value'].mean()         # BUG: Includes future data
global_std = time_series_data['value'].std()           # BUG: Includes future data
global_max = time_series_data['value'].max()           # BUG: Includes future data

# Apply global statistics as features
time_series_data['normalized_by_global'] = (time_series_data['value'] - global_mean) / global_std
time_series_data['scaled_by_global_max'] = time_series_data['value'] / global_max

# INCORRECT: Forward-looking feature engineering
# Calculate stats using future data
time_series_data['next_month_mean'] = time_series_data['value'].rolling(
    window=30, center=False
).mean().shift(-15)  # BUG: Shifts result backward, creating future leakage

# INCORRECT: Target encoding using entire time series
monthly_stats = time_series_data.groupby(time_series_data['timestamp'].dt.month)['value'].mean()
time_series_data['month_global_mean'] = time_series_data['timestamp'].dt.month.map(monthly_stats)

# Prepare features with temporal leakage
features = [
    'future_value_1', 'future_value_7', 'future_value_30',  # Future data
    'rolling_centered_7', 'rolling_centered_15',            # Centered rolling
    'normalized_by_global', 'scaled_by_global_max',         # Global stats
    'next_month_mean', 'month_global_mean',                 # Forward-looking
    'noise'
]

# Clean data
clean_data = time_series_data.dropna()
X = clean_data[features]
y = clean_data['value']

print(f"❌ Feature matrix with LEAKED features: {X.shape}")

# INCORRECT: Random split on temporal data (destroys chronological order)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # BUG: Random split on time series!
)

print("❌ Using RANDOM split on temporal data - destroys causality!")
print("❌ Training set may contain future data relative to test set")

# Train model on contaminated data
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

print(f"❌ CONTAMINATED Temporal Model - MSE: {mse:.4f}")
print("❌ TEMPORAL LEAKAGE: Model has access to future information")
print("❌ Results are invalid due to look-ahead bias")

# INCORRECT: Cross-validation that doesn't respect temporal order
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model, X, y, cv=5)  # BUG: Random CV on time series
print(f"❌ Invalid CV Score: {np.mean(cv_scores):.4f}")
print("❌ Cross-validation folds may contain future data in training set")

# INCORRECT: Feature importance analysis on leaked features
feature_importance = model.feature_importances_
feature_names = features

print("\n❌ Feature Importance (CONTAMINATED):")
for name, importance in sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True):
    print(f"   {name}: {importance:.3f}")

print("\n❌ High importance on future features indicates severe temporal leakage!")
print("❌ This model will fail catastrophically in production")