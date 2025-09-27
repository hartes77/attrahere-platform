"""
CLEAN CODE - Gestione GPU memory ottimale
EXPECTED ERRORS: 0

Questo file dimostra la gestione CORRETTA della memoria GPU:
1. Detach appropriato per metriche
2. Cache cleanup periodico
3. Context managers per evaluation
"""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

# ✅ Global seeds per garantire riproducibilità
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Setup model e data
model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1)).cuda()

optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()

# Dummy data
X = torch.randn(1000, 10).cuda()
y = torch.randn(1000, 1).cuda()
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32)

# ✅ CORRETTO: Training loop con memory management
training_losses = []

for epoch in range(10):
    epoch_losses = []

    for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
        optimizer.zero_grad()

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        # ✅ CORRETTO: Detach per storage (no gradient tracking)
        epoch_losses.append(loss.detach().cpu().item())

        # ✅ CORRETTO: Periodic cache cleanup
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()

    # ✅ CORRETTO: Average loss calculation
    avg_loss = sum(epoch_losses) / len(epoch_losses)
    training_losses.append(avg_loss)

    print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

# ✅ CORRETTO: Evaluation with no_grad context
model.eval()
with torch.no_grad():
    test_outputs = model(X[:100])  # Subset for testing
    test_loss = criterion(test_outputs, y[:100])

    # ✅ CORRETTO: Convert to scalar for logging
    final_loss = test_loss.item()

print(f"Final test loss: {final_loss:.6f}")
