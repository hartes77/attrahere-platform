"""
Clean Data Flow Examples

This file demonstrates proper data flow practices that avoid contamination
and maintain the integrity of train/test separation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

# Load sample data
data = pd.read_csv('dataset.csv')
X = data.drop('target', axis=1)
y = data['target']

# CORRECT 1: Split data FIRST, then compute statistics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compute statistics only on training data
train_mean = X_train.mean()
train_std = X_train.std()

# Apply normalization using training statistics
X_train_normalized = (X_train - train_mean) / train_std
X_test_normalized = (X_test - train_mean) / train_std

# CORRECT 2: Preprocessing after split
# Split first
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit scaler only on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)  # Only transform, don't fit

# CORRECT 3: Feature engineering with proper train/test separation
# Split data first
X_train_fe, X_test_fe, y_train_fe, y_test_fe = train_test_split(X, y, test_size=0.2, random_state=42)

# Target encoding using only training data
train_target_means = X_train_fe.groupby('category')[y_train_fe].mean()

# Apply encoding to both sets using training statistics
X_train_fe['category_encoded'] = X_train_fe['category'].map(train_target_means)
X_test_fe['category_encoded'] = X_test_fe['category'].map(train_target_means)

# Handle unseen categories in test set
X_test_fe['category_encoded'].fillna(train_target_means.mean(), inplace=True)

# CORRECT 4: Cross-validation with preprocessing inside pipeline
# Use Pipeline to ensure preprocessing happens within each CV fold
pipeline = Pipeline([
    ('encoder', LabelEncoder()),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

# CV with proper preprocessing isolation
cv_scores = cross_val_score(pipeline, X, y, cv=5)

# CORRECT 5: Temporal data flow without future leakage
time_series_data = pd.DataFrame({
    'timestamp': pd.date_range('2020-01-01', periods=1000, freq='D'),
    'value': np.random.randn(1000),
    'target': np.random.randint(0, 2, 1000)
})

# Use only past data with positive shift
time_series_data['lag_1'] = time_series_data['value'].shift(1)  # Previous value (safe)
time_series_data['lag_2'] = time_series_data['value'].shift(2)  # 2 days ago (safe)

# Rolling operations using only past data
time_series_data['rolling_mean'] = time_series_data['value'].rolling(
    window=5, center=False  # center=False ensures only past data is used
).mean()

# Temporal split for time series (chronological order)
split_point = int(0.8 * len(time_series_data))
train_data = time_series_data[:split_point]  # Past data for training
test_data = time_series_data[split_point:]   # Future data for testing

X_train_ts = train_data[['value', 'lag_1', 'lag_2', 'rolling_mean']].dropna()
y_train_ts = train_data['target'][:len(X_train_ts)]
X_test_ts = test_data[['value', 'lag_1', 'lag_2', 'rolling_mean']].dropna()
y_test_ts = test_data['target'][:len(X_test_ts)]

# CORRECT 6: Feature selection within cross-validation
# Use Pipeline to include feature selection within CV folds
feature_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(f_classif, k=10)),
    ('classifier', RandomForestClassifier())
])

cv_scores_with_selection = cross_val_score(feature_pipeline, X, y, cv=5)

# CORRECT 7: Multi-step preprocessing pipeline
# Create pipeline that handles all preprocessing steps
complete_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Will use training mean in each fold
    ('scaler', StandardScaler()),                 # Will fit on training data in each fold
    ('selector', SelectKBest(f_classif, k=5)),   # Will select on training data in each fold
    ('classifier', RandomForestClassifier())
])

# Split data properly
X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit pipeline on training data
complete_pipeline.fit(X_train_clean, y_train_clean)

# Evaluate on test data (each step uses training-only statistics)
test_score = complete_pipeline.score(X_test_clean, y_test_clean)

# Cross-validation with complete pipeline
cv_scores_complete = cross_val_score(complete_pipeline, X, y, cv=5)

# CORRECT 8: Proper handling of temporal features
def create_temporal_features(df, target_col=None, fit_on_train_only=True):
    """
    Create temporal features properly avoiding future leakage.
    """
    df_copy = df.copy()
    
    # Only use past information
    df_copy['day_of_week'] = df_copy['timestamp'].dt.dayofweek
    df_copy['month'] = df_copy['timestamp'].dt.month
    df_copy['quarter'] = df_copy['timestamp'].dt.quarter
    
    # Lag features (safe)
    for lag in [1, 7, 30]:  # 1 day, 1 week, 1 month ago
        df_copy[f'value_lag_{lag}'] = df_copy['value'].shift(lag)
    
    # Rolling features with proper window (past only)
    for window in [7, 30]:
        df_copy[f'rolling_mean_{window}'] = df_copy['value'].rolling(
            window=window, min_periods=1
        ).mean()
        df_copy[f'rolling_std_{window}'] = df_copy['value'].rolling(
            window=window, min_periods=1
        ).std()
    
    return df_copy

# Apply temporal feature engineering
ts_features = create_temporal_features(time_series_data)

# Use temporal split (not random split)
train_size = int(0.8 * len(ts_features))
ts_train = ts_features[:train_size]
ts_test = ts_features[train_size:]

print("Clean data flow examples completed")
print("This code demonstrates proper practices to avoid data leakage")