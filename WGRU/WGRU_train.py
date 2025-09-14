#基于pytorch的GRU实现代码
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


grid_x, grid_y = 4, 50
seq_len = 50
file_path = './RP_power_data.csv'
data = pd.read_csv(file_path, header=None)
x_coords = data.iloc[0::2].to_numpy(dtype=np.int64)
y_coords = data.iloc[1::2].to_numpy(dtype=np.int64)
x_hist = x_coords[:, :seq_len] - 1
y_hist = y_coords[:, :seq_len] - 1
x_next = x_coords[:, seq_len] - 1
y_next = y_coords[:, seq_len] - 1
assert x_hist.min() >= 0 and x_hist.max() < grid_x
assert y_hist.min() >= 0 and y_hist.max() < grid_y
assert x_next.min() >= 0 and x_next.max() < grid_x
assert y_next.min() >= 0 and y_next.max() < grid_y
x_hist_f = x_hist.astype(np.float32) / (grid_x - 1)
y_hist_f = y_hist.astype(np.float32) / (grid_y - 1)
X = np.stack([x_hist_f, y_hist_f], axis=-1).astype(np.float32)
y = (y_next * grid_x + x_next).astype(np.int64)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

X_train_tensor = torch.from_numpy(X_train)
y_train_tensor = torch.from_numpy(y_train)
X_test_tensor  = torch.from_numpy(X_test)
y_test_tensor  = torch.from_numpy(y_test)

class GRUModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_classes=200, num_layers=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        logits = self.fc(last)
        return logits

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GRUModel(input_size=2, hidden_size=128, num_classes=grid_x*grid_y,
                 num_layers=1, dropout=0.0).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 50
batch_size = 32
num_train = X_train_tensor.size(0)
best_state = None
best_loss = float('inf')
for epoch in range(1, epochs + 1):
    model.train()
    perm = torch.randperm(num_train)
    epoch_loss = 0.0
    n_batches = 0

    for i in range(0, num_train, batch_size):
        idx = perm[i:i+batch_size]
        batch_x = X_train_tensor[idx].to(device)
        batch_y = y_train_tensor[idx].to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    avg_loss = epoch_loss / max(1, n_batches)
    print(f"Epoch {epoch}/{epochs} | Avg Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

os.makedirs('./checkpoints', exist_ok=True)
torch.save(model.state_dict(), './checkpoints/gru_last.pth')
if best_state is not None:
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), './checkpoints/gru_best.pth')
print("Saved to ./checkpoints/gru_last.pth and ./checkpoints/gru_best.pth")
model.eval()
with torch.no_grad():
    logits_test = model(X_test_tensor.to(device))
    pred = torch.argmax(logits_test, dim=1).cpu()
    acc = (pred == y_test_tensor).float().mean().item()
    print(f"Test Accuracy: {acc*100:.2f}%")
plt.figure(figsize=(10, 6))
plt.plot(y_test[:100], label="True Position IDs")
plt.plot(pred.numpy()[:100], label="Predicted Position IDs", linestyle='dashed')
plt.legend()
plt.title("GRU: True vs Predicted Position IDs (First 100 Points)")
plt.tight_layout()
plt.show()
