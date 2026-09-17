import torch
import torch.nn as nn
import numpy as np

import matplotlib

matplotlib.use("Agg")  # Headless HPC backend
import matplotlib.pyplot as plt

# =====================================================================
# 1. Load Data and Pre-Trained Model
# =====================================================================
data = torch.load("dataset.pt")
X_test_t, y_test_t = data["X_test"], data["y_test"]
batch_size = X_test_t.shape[0]  # 360 reference samples


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
model.load_state_dict(torch.load("mlp_model.pth"))
model.eval()

with torch.no_grad():
    orig_test_preds = model(X_test_t).argmax(dim=1)

# =====================================================================
# 2. Calculate Causal Scores
# =====================================================================
scores_l1 = np.zeros(256)
scores_l2 = np.zeros(128)

with torch.no_grad():
    for i in range(256):
        h1 = model.relu1(model.fc1(X_test_t))
        h1[:, i] = 0.0
        h2 = model.relu2(model.fc2(h1))
        scores_l1[i] = (
            (orig_test_preds != model.fc3(h2).argmax(dim=1)).float().mean().item()
        )

    for j in range(128):
        h1 = model.relu1(model.fc1(X_test_t))
        h2 = model.relu2(model.fc2(h1))
        h2[:, j] = 0.0
        scores_l2[j] = (
            (orig_test_preds != model.fc3(h2).argmax(dim=1)).float().mean().item()
        )

all_scores = np.concatenate([scores_l1, scores_l2])

# =====================================================================
# 3. Grid Search Over Retention % Using Exact Rank-Based Top-K
# =====================================================================
retention_percentages = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
accuracies = []
agreements = []
batch_mflops = []
retained_counts = []

print("=== GRID SEARCH STUDY: RETENTION % vs ACCURACY, AGREEMENT & FLOPs ===")
print(
    f"{'Retention %':<12} | {'Kept Neurons':<12} | {'Test Acc %':<12} | {'Agreement %':<12} | {'Batch MFLOPs':<12}"
)
print("-" * 72)

for pct in retention_percentages:
    # 1. Enforce exact K neuron selection using rank sorting
    k = max(1, int(round((pct / 100.0) * len(all_scores))))
    top_k_indices = set(np.argsort(all_scores)[-k:])

    # 2. Extract separate layer masks
    mask_l1 = np.array([i in top_k_indices for i in range(256)])
    mask_l2 = np.array([(j + 256) in top_k_indices for j in range(128)])

    n_h1 = int(mask_l1.sum())
    n_h2 = int(mask_l2.sum())
    total_retained = n_h1 + n_h2

    # 3. Multiply-Accumulate FLOPs: 2 * in_dim * out_dim per layer
    flops_l1 = 2 * 64 * n_h1
    flops_l2 = 2 * n_h1 * n_h2
    flops_l3 = 2 * n_h2 * 10

    flops_per_sample = flops_l1 + flops_l2 + flops_l3
    total_mflops = (flops_per_sample * batch_size) / 1e6  # Convert to MFLOPs per batch

    # 4. Evaluate network accuracy & agreement
    with torch.no_grad():
        if n_h1 == 0 or n_h2 == 0:
            acc = 10.0
            agree = 10.0
        else:
            W1 = model.fc1.weight[mask_l1, :]
            b1 = model.fc1.bias[mask_l1]

            W2 = model.fc2.weight[mask_l2, :][:, mask_l1]
            b2 = model.fc2.bias[mask_l2]

            W3 = model.fc3.weight[:, mask_l2]
            b3 = model.fc3.bias

            h1_p = torch.relu(torch.matmul(X_test_t, W1.T) + b1)
            h2_p = torch.relu(torch.matmul(h1_p, W2.T) + b2)
            p_logits = torch.matmul(h2_p, W3.T) + b3
            p_preds = p_logits.argmax(dim=1)

            acc = (p_preds == y_test_t).float().mean().item() * 100
            agree = (p_preds == orig_test_preds).float().mean().item() * 100

    accuracies.append(acc)
    agreements.append(agree)
    batch_mflops.append(total_mflops)
    retained_counts.append(total_retained)

    print(
        f"{pct:<12}% | {total_retained:<12} | {acc:<12.2f}% | {agree:<12.2f}% | {total_mflops:<12.3f} MFLOPs"
    )

# =====================================================================
# 4. Save Dual-Axis Trade-off Plot
# =====================================================================
fig, ax1 = plt.subplots(figsize=(11, 6))

# Primary Y-Axis: Accuracy & Agreement (%)
color_acc = "tab:blue"
color_agree = "tab:orange"
line1 = ax1.plot(
    retention_percentages,
    accuracies,
    "o-",
    color=color_acc,
    linewidth=2,
    label="Test Accuracy (%)",
)
line2 = ax1.plot(
    retention_percentages,
    agreements,
    "s--",
    color=color_agree,
    linewidth=2,
    label="Prediction Agreement (%)",
)
ax1.set_xlabel("Neuron Retention Percentage (%)", fontsize=12)
ax1.set_ylabel("Accuracy / Agreement (%)", fontsize=12)
ax1.grid(True, linestyle="--", alpha=0.5)

# Secondary Y-Axis: FLOPs (MFLOPs per batch)
ax2 = ax1.twinx()
color_flops = "tab:red"
line3 = ax2.plot(
    retention_percentages,
    batch_mflops,
    "d-.",
    color=color_flops,
    linewidth=2,
    label="Batch FLOPs (MFLOPs)",
)
ax2.set_ylabel("Computation (MFLOPs per Batch)", color=color_flops, fontsize=12)
ax2.tick_params(axis="y", labelcolor=color_flops)

# Merge Legends
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="center right", fontsize=10)

plt.title(
    "Trade-off Study: Retention % vs Accuracy, Agreement & Computational FLOPs",
    fontsize=14,
)
plt.tight_layout()

plt.savefig("retention_grid_search.png", dpi=300)
plt.close()
print("\nPlot saved: 'retention_grid_search.png'")
