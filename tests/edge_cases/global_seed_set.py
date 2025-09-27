"""
EDGE CASE - Global seed dovrebbe sopprimere warning locali
EXPECTED ERRORS: 0

Questo è un test di intelligenza del sistema:
Quando global seed è impostato, non dovrebbe segnalare errori
su funzioni numpy random senza random_state locale.
"""

import random

import numpy as np
from sklearn.utils import shuffle

# ✅ Global seed impostato correttamente all'inizio
np.random.seed(42)
random.seed(42)

# Dataset
X = np.random.randn(1000, 20)  # ✅ OK - global seed copre questo
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# ✅ Queste operazioni DOVREBBERO essere OK perché global seed è set
data_shuffled = shuffle(X, y)  # NO random_state needed - global is set
random_sample = np.random.choice(len(X), 100)  # OK - global seed active

# ✅ Test intelligenza del sistema: deve riconoscere che il global seed
# rende deterministici questi operations
print("Global seed makes everything deterministic!")
print(f"First random number: {np.random.rand()}")  # Sempre lo stesso
