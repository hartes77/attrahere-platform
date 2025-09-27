"""
CLEAN CODE - Reproducibilità perfetta
EXPECTED ERRORS: 0

Questo file dimostra la configurazione CORRETTA per reproducibilità:
1. Global seeds impostati all'inizio
2. Random states specificati dove necessario
3. Risultati completamente deterministici
"""

import random

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ✅ CORRETTO: Global seeds all'inizio del file
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ✅ CORRETTO: Dataset generation deterministico
X = np.random.randn(1000, 20)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# ✅ CORRETTO: Split con random_state esplicito (anche se global è set)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
)

# ✅ CORRETTO: Model con random_state
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,  # Sempre meglio essere espliciti
    n_jobs=1,  # Deterministic per evitare racing conditions
)

# ✅ CORRETTO: Training deterministico
model.fit(X_train, y_train)
score = model.score(X_test, y_test)

print(f"Test accuracy: {score:.6f}")  # Risultato sempre identico
