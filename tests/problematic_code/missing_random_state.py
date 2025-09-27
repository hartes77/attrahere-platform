"""
PROBLEMATIC CODE - Non-reproducibilità
EXPECTED ERRORS: 1 (missing_random_seed)

Questo file contiene ESATTAMENTE UN ERRORE:
train_test_split() senza random_state parameter.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Setup corretto
np.random.seed(42)  # Global seed impostato

# Dataset generation
X = np.random.randn(1000, 20)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# ❌ ERROR: Missing random_state - risultati non riproducibili!
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,  # ← ERRORE QUI: manca random_state=42
)

# Il resto è corretto
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)

print(f"Non-reproducible accuracy: {score:.3f}")
# ^ Questo valore cambierà ad ogni run!
