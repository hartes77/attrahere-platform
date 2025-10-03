"""
Scaler Incorrect Implementation - Validation Suite

This file demonstrates the INCORRECT way to apply preprocessing:
- Fit scaler on ENTIRE dataset (including test data)
- Split data AFTER preprocessing
- This causes data leakage and should be detected by our detector
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load sample data
np.random.seed(42)
data = pd.DataFrame({
    'feature1': np.random.normal(10, 2, 1000),
    'feature2': np.random.normal(5, 1.5, 1000),
    'feature3': np.random.normal(0, 1, 1000),
    'target': np.random.randint(0, 2, 1000)
})

X = data.drop('target', axis=1)
y = data['target']

# INCORRECT: Fit scaler on ENTIRE dataset (including future test data)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # BUG: Uses test set information in fitting!

# INCORRECT: Split AFTER preprocessing (too late!)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model on contaminated data
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate (results may be overly optimistic due to leakage)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"❌ INCORRECT Implementation - Accuracy: {accuracy:.4f}")
print("❌ DATA LEAKAGE: scaler fitted on entire dataset including test data")
print("❌ Test set information used in preprocessing - results are contaminated")