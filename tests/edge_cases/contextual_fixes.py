"""
EDGE CASE - Fix applicati in scope diverso
EXPECTED ERRORS: 0

Test di intelligenza: il sistema deve riconoscere che il fix
è applicato in uno scope che influenza l'operazione problematica.
"""

import random

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ✅ Global seeds per isolare il test al solo contextual fixes
np.random.seed(42)
random.seed(42)


def prepare_data():
    """Function che prepara i dati correttamente"""
    # Dataset
    X = np.random.randn(1000, 20)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # ✅ CORRETTO: Split prima di preprocessing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )

    return X_train, X_test, y_train, y_test


def preprocess_data(X_train, X_test):
    """Function separata per preprocessing - ma CORRETTA"""
    # ✅ CORRETTO: Fit solo su training, poi transform su entrambi
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


# ✅ Il sistema deve riconoscere che questo workflow è CORRETTO
# anche se distribuito su multiple functions
X_train, X_test, y_train, y_test = prepare_data()
X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test)

print("Contextual workflow completed correctly!")
