"""
PROBLEMATIC CODE - GPU Memory Leak
EXPECTED ERRORS: 1 (gpu_memory_leak)

Questo file contiene ESATTAMENTE UN ERRORE:
Accumulo di tensori con gradienti in un loop senza .detach()
"""

import torch
from torch import nn

# ✅ Global seeds per isolare il test al solo GPU memory leak
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Setup
model = nn.Linear(10, 1)
criterion = nn.MSELoss()

# Training data
X = torch.randn(100, 10)
y = torch.randn(100, 1)

# ❌ CRITICAL ERROR: Tensor accumulation causing memory leak
losses = []

for i in range(50):  # Simulation di training loop
    output = model(X)
    loss = criterion(output, y)

    # ❌ ERRORE QUI: Accumulating tensor with gradients!
    losses.append(loss)  # ← MEMORY LEAK! Dovrebbe essere loss.detach()

    # Il resto è corretto
    loss.backward()

# Il gradient computation graph cresce indefinitamente
# GPU memory si esaurisce rapidamente
print(f"Collected {len(losses)} losses")
print(f"Final loss: {losses[-1].item():.6f}")
