"""
CLEAN CODE - Preprocessing corretto dopo train/test split
EXPECTED ERRORS: 0

Questo file dimostra il modo CORRETTO di fare preprocessing:
1. Prima split dei dati
2. Poi fit del preprocessor SOLO sui training data
3. Transform su entrambi train e test
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Carica dataset
data = pd.read_csv("dataset.csv")
X = data.drop("target", axis=1)
y = data["target"]

# ✅ CORRETTO: Prima split, poi preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
)

# ✅ CORRETTO: Fit solo sui training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit + transform training
X_test_scaled = scaler.transform(X_test)  # Solo transform test

# ✅ CORRETTO: Training con dati preprocessati correttamente
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# ✅ CORRETTO: Valutazione pulita
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.3f}")
