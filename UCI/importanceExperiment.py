import torch
import torch.nn as nn
import numpy as np

import matplotlib

matplotlib.use("Agg")  # Headless HPC backend
import matplotlib.pyplot as plt

# =====================================================================
# 1. Load Data and Pre-Trained Baseline Model
# =====================================================================
data = torch.load("dataset.pt")
X_test_t, y_test_t = data["X_test"], data["y_test"]


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
# 2. Compute Exact Causal Importance Scores
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


# Helper function to evaluate model given layer masks
def evaluate_masked_model(mask_l1, mask_l2):
    n_h1, n_h2 = int(mask_l1.sum()), int(mask_l2.sum())
    if n_h1 == 0 or n_h2 == 0:
        return 10.0, 10.0

    with torch.no_grad():
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
    return acc, agree


# =====================================================================
# 3. Experiment: Causal vs Random vs Bottom-K Selection
# =====================================================================
retention_percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
num_random_trials = 10

causal_accs, causal_agrees = [], []
bottom_accs, bottom_agrees = [], []
random_acc_means, random_acc_stds = [], []
random_agree_means, random_agree_stds = [], []

ranked_indices = np.argsort(all_scores)  # Ascending order (lowest score first)

print("=== CANCELLATION STUDY: TOP-K CAUSAL vs RANDOM vs BOTTOM-K ===")
print(
    f"{'Retention %':<12} | {'Causal Acc %':<14} | {'Random Acc % (Mean±Std)':<24} | {'Bottom-K Acc %':<14}"
)
print("-" * 70)

for pct in retention_percentages:
    k = max(1, int(round((pct / 100.0) * len(all_scores))))

    # 1. Top-K Causal Selection (Most Important)
    top_k_indices = set(ranked_indices[-k:])
    mask_l1_top = np.array([i in top_k_indices for i in range(256)])
    mask_l2_top = np.array([(j + 256) in top_k_indices for j in range(128)])
    c_acc, c_agree = evaluate_masked_model(mask_l1_top, mask_l2_top)
    causal_accs.append(c_acc)
    causal_agrees.append(c_agree)

    # 2. Bottom-K Selection (Least Important)
    bottom_k_indices = set(ranked_indices[:k])
    mask_l1_bot = np.array([i in bottom_k_indices for i in range(256)])
    mask_l2_bot = np.array([(j + 256) in bottom_k_indices for j in range(128)])
    b_acc, b_agree = evaluate_masked_model(mask_l1_bot, mask_l2_bot)
    bottom_accs.append(b_acc)
    bottom_agrees.append(b_agree)

    # 3. Random Selection (Averaged over N trials)
    r_accs_trial, r_agrees_trial = [], []
    for seed in range(num_random_trials):
        np.random.seed(seed + 100)
        rand_indices = set(np.random.choice(384, size=k, replace=False))
        mask_l1_rand = np.array([i in rand_indices for i in range(256)])
        mask_l2_rand = np.array([(j + 256) in rand_indices for j in range(128)])
        r_a, r_g = evaluate_masked_model(mask_l1_rand, mask_l2_rand)
        r_accs_trial.append(r_a)
        r_agrees_trial.append(r_g)

    r_acc_mean, r_acc_std = np.mean(r_accs_trial), np.std(r_accs_trial)
    random_acc_means.append(r_acc_mean)
    random_acc_stds.append(r_acc_std)
    random_agree_means.append(np.mean(r_agrees_trial))
    random_agree_stds.append(np.std(r_agrees_trial))

    print(
        f"{pct:<12}% | {c_acc:<14.2f}% | {r_acc_mean:5.2f}% ± {r_acc_std:<13.2f} | {b_acc:<14.2f}%"
    )

# =====================================================================
# 4. Save Plot
# =====================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Plot Curves
ax.plot(
    retention_percentages,
    causal_accs,
    "o-",
    color="tab:green",
    linewidth=2.5,
    label="Top-K Causal (Most Important)",
)
ax.errorbar(
    retention_percentages,
    random_acc_means,
    yerr=random_acc_stds,
    fmt="s--",
    color="tab:blue",
    linewidth=2,
    capsize=4,
    label=f"Random Selection (Mean ± Std, n={num_random_trials})",
)
ax.plot(
    retention_percentages,
    bottom_accs,
    "x-.",
    color="tab:red",
    linewidth=2,
    label="Bottom-K Causal (Least Important)",
)

ax.set_xlabel("Neuron Retention Percentage (%)", fontsize=12)
ax.set_ylabel("Test Accuracy (%)", fontsize=12)
ax.set_title(
    "Ablation Benchmark: Causal vs Random vs Bottom-K Neuron Selection", fontsize=14
)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(fontsize=11)
plt.tight_layout()

plt.savefig("causal_vs_random_pruning.png", dpi=300)
plt.close()
print("\nPlot saved: 'causal_vs_random_pruning.png'")
