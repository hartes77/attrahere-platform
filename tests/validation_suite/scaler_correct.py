"""
Scaler Correct Implementation - Validation Suite

This file demonstrates the CORRECT way to apply preprocessing:
- Split data FIRST
- Fit scaler only on training data
- Transform both training and test data using training statistics
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

# CORRECT: Split data FIRST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CORRECT: Fit scaler only on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit AND transform training data
X_test_scaled = scaler.transform(X_test)        # Only transform test data (no fitting)

# Train model on properly scaled data
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate on properly scaled test data
predictions = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)

print(f"✅ CORRECT Implementation - Accuracy: {accuracy:.4f}")
print("✅ No data leakage: scaler fitted only on training data")
print("✅ Test set information never used in preprocessing")