import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
grid_x, grid_y = 4, 50
num_states = grid_x * grid_y
seq_len = 50
file_path = './RP_power_data.csv'
data = pd.read_csv(file_path, header=None)
x_coords = data.iloc[0::2].to_numpy(dtype=np.int64)
y_coords = data.iloc[1::2].to_numpy(dtype=np.int64)
x_all = x_coords - 1
y_all = y_coords - 1
assert x_all.min() >= 0 and x_all.max() < grid_x
assert y_all.min() >= 0 and y_all.max() < grid_y
states_all = y_all * grid_x + x_all
states_train, states_test = train_test_split(states_all, test_size=0.2, shuffle=False)
pi_counts = np.zeros(num_states, dtype=np.float64)
for s in states_train[:, 0]:
    pi_counts[s] += 1.0

A_counts = np.zeros((num_states, num_states), dtype=np.float64)
for seq in states_train:
    for t in range(seq_len):
        i = seq[t]
        j = seq[t + 1]
        A_counts[i, j] += 1.0

alpha = 1.0
pi = (pi_counts + alpha) / (pi_counts.sum() + alpha * num_states)
A = (A_counts + alpha)
A = A / A.sum(axis=1, keepdims=True)

y_true = states_test[:, seq_len]
last_obs = states_test[:, seq_len - 1]
y_pred = np.empty_like(y_true)
for k, i in enumerate(last_obs):
    y_pred[k] = np.argmax(A[i])

acc = (y_pred == y_true).mean()
print(f"HMM Top-1 Accuracy: {acc * 100:.2f}%")

# Optional: Top-k (e.g., Top-3)
k = 3
topk = np.argsort(-A[last_obs], axis=1)[:, :k]
topk_hit = np.any(topk == y_true[:, None], axis=1).mean()
print(f"HMM Top-{k} Accuracy: {topk_hit * 100:.2f}%")
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(y_true[:100], label='True State IDs')
    plt.plot(y_pred[:100], label='Predicted State IDs', linestyle='--')
    plt.title('HMM: True vs Predicted State IDs (First 100 Points)')
    plt.legend()
    plt.tight_layout()
    plt.show()
except Exception:
    pass
