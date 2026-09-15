import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Configure non-interactive backend for HPC clusters
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# 1. Dataset Setup (1437 train, 360 test/reference samples)
# =====================================================================
digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=360, random_state=42, stratify=y
)

mean, std = X_train.mean(axis=0, keepdims=True), X_train.std(axis=0, keepdims=True) + 1e-8
X_train_std = (X_train - mean) / std
X_test_std = (X_test - mean) / std

X_train_t = torch.tensor(X_train_std, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test_std, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# =====================================================================
# Model Definition (256x128 = 384 hidden neurons)
# =====================================================================
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        h1 = self.relu1(self.fc1(x))
        h2 = self.relu2(self.fc2(h1))
        out = self.fc3(h2)
        return out

model = MLP()
optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()

# Train original network
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    loss = criterion(model(X_train_t), y_train_t)
    loss.backward()
    optimizer.step()

model.eval()

# =====================================================================
# 1. Baseline Model Performance
# =====================================================================
def get_accuracy(model_obj, X_t, y_t):
    with torch.no_grad():
        preds = model_obj(X_t).argmax(dim=1)
        acc = (preds == y_t).float().mean().item()
    return acc

baseline_train_acc = get_accuracy(model, X_train_t, y_train_t)
baseline_test_acc = get_accuracy(model, X_test_t, y_test_t)

print("=== 1. BASELINE MODEL PERFORMANCE ===")
print(f"Training Accuracy: {baseline_train_acc * 100:.2f}%")
print(f"Test Accuracy:     {baseline_test_acc * 100:.2f}%\n")

with torch.no_grad():
    orig_test_preds = model(X_test_t).argmax(dim=1)

# =====================================================================
# Causal Ablation Scoring (1 score per neuron)
# =====================================================================
scores_l1 = np.zeros(256)
scores_l2 = np.zeros(128)

with torch.no_grad():
    for i in range(256):
        h1 = model.relu1(model.fc1(X_test_t))
        h1[:, i] = 0.0
        h2 = model.relu2(model.fc2(h1))
        ablated_preds = model.fc3(h2).argmax(dim=1)
        scores_l1[i] = (orig_test_preds != ablated_preds).float().mean().item()

    for j in range(128):
        h1 = model.relu1(model.fc1(X_test_t))
        h2 = model.relu2(model.fc2(h1))
        h2[:, j] = 0.0
        ablated_preds = model.fc3(h2).argmax(dim=1)
        scores_l2[j] = (orig_test_preds != ablated_preds).float().mean().item()

# =====================================================================
# 2. Save Heatmaps per Layer to File
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im1 = axes[0].imshow(scores_l1.reshape(16, 16), cmap='viridis')
axes[0].set_title("Layer 1 Neuron Causal Scores (16x16)")
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(scores_l2.reshape(8, 16), cmap='viridis')
axes[1].set_title("Layer 2 Neuron Causal Scores (8x16)")
plt.colorbar(im2, ax=axes[1])

plt.suptitle("Neuron Causal Importance Scores", fontsize=14)
plt.tight_layout()

# Save plot instead of showing
plt.savefig('neuron_causal_scores.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved plot: neuron_causal_scores.png")

# =====================================================================
# 3. Save Retained Neurons Mask to File
# =====================================================================
all_scores = np.concatenate([scores_l1, scores_l2])
cutoff = np.percentile(all_scores, 85)

mask_l1 = (scores_l1 >= cutoff).astype(int)
mask_l2 = (scores_l2 >= cutoff).astype(int)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(mask_l1.reshape(16, 16), cmap='Blues', vmin=0, vmax=1)
axes[0].set_title(f"Layer 1 Retained Neurons ({mask_l1.sum()}/256)")

axes[1].imshow(mask_l2.reshape(8, 16), cmap='Blues', vmin=0, vmax=1)
axes[1].set_title(f"Layer 2 Retained Neurons ({mask_l2.sum()}/128)")

plt.suptitle("Retained Neurons Mask (1 = Kept, 0 = Pruned)", fontsize=14)
plt.tight_layout()

# Save plot instead of showing
plt.savefig('retained_neurons_mask.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved plot: retained_neurons_mask.png\n")

# =====================================================================
# 4. Model Accuracy After Pruning
# =====================================================================
with torch.no_grad():
    W1 = model.fc1.weight.clone()
    b1 = model.fc1.bias.clone()
    W2 = model.fc2.weight.clone()
    b2 = model.fc2.bias.clone()
    W3 = model.fc3.weight.clone()
    b3 = model.fc3.bias.clone()

    for i in range(256):
        if mask_l1[i] == 0:
            W1[i, :] = 0.0
            b1[i] = 0.0

    for j in range(128):
        if mask_l2[j] == 0:
            W2[j, :] = 0.0
            b2[j] = 0.0

    h1_pruned = torch.relu(torch.matmul(X_test_t, W1.T) + b1)
    h2_pruned = torch.relu(torch.matmul(h1_pruned, W2.T) + b2)
    pruned_logits = torch.matmul(h2_pruned, W3.T) + b3
    pruned_preds = pruned_logits.argmax(dim=1)

    pruned_test_acc = (pruned_preds == y_test_t).float().mean().item()
    pruned_agreement = (pruned_preds == orig_test_preds).float().mean().item()

print("=== 2. POST-PRUNING PERFORMANCE (Top 15% Retained) ===")
print(f"Total Retained Neurons: {mask_l1.sum() + mask_l2.sum()} / 384")
print(f"Pruned Model Accuracy:  {pruned_test_acc * 100:.2f}%")
print(f"Prediction Agreement:   {pruned_agreement * 100:.2f}%")
