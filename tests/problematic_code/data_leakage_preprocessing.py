"""
PROBLEMATIC CODE - Data leakage critico
EXPECTED ERRORS: 1 (data_leakage_preprocessing)

Questo file contiene ESATTAMENTE UN ERRORE critico:
StandardScaler.fit_transform() applicato all'intero dataset prima del train_test_split.
"""

import random

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ✅ Global seeds per isolare il test al solo data leakage
np.random.seed(42)
random.seed(42)

# Dataset generation
X = np.random.randn(1000, 20)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# ❌ CRITICAL ERROR: Preprocessing prima di split - DATA LEAKAGE!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # ← ERRORE QUI: fit su tutto il dataset

# Split dopo preprocessing - troppo tardi!
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42,
)

# Il resto è corretto ma inutile - il danno è già fatto
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Le performance saranno falsamente ottimistiche
score = model.score(X_test, y_test)
print(f"Overly optimistic accuracy: {score:.3f}")
