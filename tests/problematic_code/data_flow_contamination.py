"""
Test case for Data Flow Contamination Detection

This file contains various patterns of data flow contamination that should be
detected by the DataFlowContaminationDetector. These patterns cause subtle
data leakage through improper pipeline ordering and global statistics usage.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Load sample data
data = pd.read_csv('dataset.csv')
X = data.drop('target', axis=1)
y = data['target']

# PROBLEM 1: Global statistics computed before split (HIGH SEVERITY)
# This causes data leakage because test set information is used in normalization
global_mean = X.mean()  # BUG: Uses entire dataset including test set
global_std = X.std()    # BUG: Uses entire dataset including test set

# Later split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply normalization using global statistics (CONTAMINATED)
X_train_normalized = (X_train - global_mean) / global_std
X_test_normalized = (X_test - global_mean) / global_std

# PROBLEM 2: Preprocessing applied to entire dataset before split (HIGH SEVERITY)
# Fit scaler on entire dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # BUG: Fits on entire dataset including test set

# Then split the already-processed data
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# PROBLEM 3: Feature engineering using global information (MEDIUM SEVERITY)
# Target encoding using entire dataset
target_means = X.groupby('category')['target'].mean()  # BUG: Uses test set targets
X['category_encoded'] = X['category'].map(target_means)

# Then split
X_train_fe, X_test_fe, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# PROBLEM 4: Cross-validation with preprocessing outside folds (HIGH SEVERITY)
# Apply preprocessing outside CV
encoder = LabelEncoder()
X_categorical_encoded = encoder.fit_transform(X['category'])  # BUG: Fits on entire dataset

# Then use in cross-validation
model = RandomForestClassifier()
cv_scores = cross_val_score(model, X_categorical_encoded, y, cv=5)  # CONTAMINATED

# PROBLEM 5: Temporal data flow contamination (HIGH SEVERITY)
# For time series data, using future information
time_series_data = pd.DataFrame({
    'timestamp': pd.date_range('2020-01-01', periods=1000, freq='D'),
    'value': np.random.randn(1000),
    'target': np.random.randint(0, 2, 1000)
})

# BUG: Using future data with negative shift
time_series_data['future_value'] = time_series_data['value'].shift(-1)  # FUTURE LEAKAGE!

# BUG: Rolling with center=True uses future data
time_series_data['rolling_centered'] = time_series_data['value'].rolling(
    window=5, center=True
).mean()  # FUTURE LEAKAGE!

# Random split on temporal data (should use temporal split)
X_ts = time_series_data[['value', 'future_value', 'rolling_centered']]
y_ts = time_series_data['target']
X_train_ts, X_test_ts, y_train_ts, y_test_ts = train_test_split(
    X_ts, y_ts, test_size=0.2, random_state=42  # BUG: Random split on temporal data
)

# PROBLEM 6: Feature selection before cross-validation (MEDIUM SEVERITY)
from sklearn.feature_selection import SelectKBest, f_classif

# BUG: Feature selection on entire dataset
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)  # Uses entire dataset including test set

# Then use in cross-validation
cv_scores_selected = cross_val_score(
    RandomForestClassifier(), X_selected, y, cv=5  # CONTAMINATED
)

# PROBLEM 7: Pipeline contamination in multiple steps (HIGH SEVERITY)
# Multiple preprocessing steps applied to entire dataset
# Step 1: Fill missing values using global statistics
X_filled = X.fillna(X.mean())  # BUG: Uses global mean including test set

# Step 2: Normalize using global statistics  
X_normalized = (X_filled - X_filled.mean()) / X_filled.std()  # BUG: Global stats

# Step 3: Feature selection on entire dataset
from sklearn.feature_selection import SelectKBest
selector = SelectKBest(k=5)
X_selected = selector.fit_transform(X_normalized, y)  # BUG: Entire dataset

# Finally split (too late!)
X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

print("Data flow contamination examples completed")
print("This code contains multiple data leakage patterns that should be detected")