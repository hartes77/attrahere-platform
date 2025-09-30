# Test case for hardcoded thresholds - PROBLEMATIC CODE
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']

# Split data properly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Get predictions
y_proba = model.predict_proba(X_test)[:, 1]

# ❌ HARDCODED THRESHOLD - No justification or analysis
threshold = 0.73625  # Magic number without explanation
y_pred = (y_proba >= threshold).astype(int)

# ❌ Another hardcoded threshold
anomaly_threshold = 0.05  # Why 0.05? No business logic documented

# ❌ Hardcoded confidence levels
if y_proba.max() > 0.9847:  # Arbitrary precision threshold
    confidence = "high"
elif y_proba.max() > 0.6123:  # Another magic number
    confidence = "medium"
else:
    confidence = "low"

# ❌ Performance threshold without validation
performance_cutoff = 0.8234567  # Overly precise, no context
if model.score(X_test, y_test) > performance_cutoff:
    print("Model accepted")