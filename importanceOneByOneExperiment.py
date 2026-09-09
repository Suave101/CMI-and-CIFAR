import torch
import torch.nn as nn
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Headless HPC backend
import matplotlib.pyplot as plt

# =====================================================================
# 1. Load Data and Pre-Trained Baseline Model
# =====================================================================
data = torch.load('dataset.pt')
X_test_t, y_test_t = data['X_test'], data['y_test']

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
model.load_state_dict(torch.load('mlp_model.pth'))
model.eval()

with torch.no_grad():
    orig_test_preds = model(X_test_t).argmax(dim=1)

# =====================================================================
# 2. Evaluation and Score Computation Helpers
# =====================================================================
def evaluate_masked_model(mask_l1, mask_l2):
    n1, n2 = int(mask_l1.sum()), int(mask_l2.sum())
    if n1 == 0 or n2 == 0:
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

def compute_causal_scores(mask_l1, mask_l2):
    """Computes causal importance scores ONLY for currently active neurons."""
    scores_l1 = np.full(256, -1.0)  # Inactive neurons receive -1.0
    scores_l2 = np.full(128, -1.0)
    
    active_l1_indices = np.where(mask_l1)[0]
    active_l2_indices = np.where(mask_l2)[0]

    with torch.no_grad():
        W1 = model.fc1.weight[mask_l1, :]
        b1 = model.fc1.bias[mask_l1]
        W2 = model.fc2.weight[mask_l2, :][:, mask_l1]
        b2 = model.fc2.bias[mask_l2]
        W3 = model.fc3.weight[:, mask_l2]
        b3 = model.fc3.bias

        h1_curr = torch.relu(torch.matmul(X_test_t, W1.T) + b1)
        h2_curr = torch.relu(torch.matmul(h1_curr, W2.T) + b2)

        # Score Layer 1 active units
        for idx_in_sub, global_idx in enumerate(active_l1_indices):
            h1_temp = h1_curr.clone()
            h1_temp[:, idx_in_sub] = 0.0
            h2_temp = torch.relu(torch.matmul(h1_temp, W2.T) + b2)
            preds = (torch.matmul(h2_temp, W3.T) + b3).argmax(dim=1)
            scores_l1[global_idx] = (orig_test_preds != preds).float().mean().item()

        # Score Layer 2 active units
        for idx_in_sub, global_idx in enumerate(active_l2_indices):
            h2_temp = h2_curr.clone()
            h2_temp[:, idx_in_sub] = 0.0
            preds = (torch.matmul(h2_temp, W3.T) + b3).argmax(dim=1)
            scores_l2[global_idx] = (orig_test_preds != preds).float().mean().item()

    return scores_l1, scores_l2

# =====================================================================
# 3. Comparative Pruning Benchmark
# =====================================================================
retention_percentages = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
num_random_trials = 10

iterative_accs = []
oneshot_accs = []
random_acc_means, random_acc_stds = [], []

print("=== PRUNING BENCHMARK: ITERATIVE vs ONE-SHOT vs RANDOM ===")
print(f"{'Retention %':<12} | {'Iterative Acc %':<18} | {'One-Shot Acc %':<18} | {'Random Acc % (Mean±Std)':<24}")
print("-" * 78)

# Initial state for iterative pruning
iter_mask_l1 = np.ones(256, dtype=bool)
iter_mask_l2 = np.ones(128, dtype=bool)

# Compute initial baseline scores once for One-Shot comparison
init_scores_l1, init_scores_l2 = compute_causal_scores(iter_mask_l1, iter_mask_l2)

for pct in retention_percentages:
    k1 = max(1, int(round((pct / 100.0) * 256)))
    k2 = max(1, int(round((pct / 100.0) * 128)))

    # 1. Iterative Causal Pruning (Recalculates scores on active subset)
    if pct == 100:
        it_acc, _ = evaluate_masked_model(iter_mask_l1, iter_mask_l2)
    else:
        curr_s1, curr_s2 = compute_causal_scores(iter_mask_l1, iter_mask_l2)
        top_l1_iter = set(np.argsort(curr_s1)[-k1:])
        top_l2_iter = set(np.argsort(curr_s2)[-k2:])
        iter_mask_l1 = np.array([i in top_l1_iter for i in range(256)])
        iter_mask_l2 = np.array([j in top_l2_iter for j in range(128)])
        it_acc, _ = evaluate_masked_model(iter_mask_l1, iter_mask_l2)
    iterative_accs.append(it_acc)

    # 2. One-Shot Layer-Wise Causal Pruning (Fixed initial scores)
    top_l1_os = set(np.argsort(init_scores_l1)[-k1:])
    top_l2_os = set(np.argsort(init_scores_l2)[-k2:])
    os_mask_l1 = np.array([i in top_l1_os for i in range(256)])
    os_mask_l2 = np.array([j in top_l2_os for j in range(128)])
    os_acc, _ = evaluate_masked_model(os_mask_l1, os_mask_l2)
    oneshot_accs.append(os_acc)

    # 3. Layer-Wise Proportional Random Selection
    r_accs_trial = []
    for seed in range(num_random_trials):
        np.random.seed(seed + 100)
        rand_l1 = set(np.random.choice(256, size=k1, replace=False))
        rand_l2 = set(np.random.choice(128, size=k2, replace=False))
        m_l1 = np.array([i in rand_l1 for i in range(256)])
        m_l2 = np.array([j in rand_l2 for j in range(128)])
        ra, _ = evaluate_masked_model(m_l1, m_l2)
        r_accs_trial.append(ra)

    r_mean, r_std = np.mean(r_accs_trial), np.std(r_accs_trial)
    random_acc_means.append(r_mean)
    random_acc_stds.append(r_std)

    print(f"{pct:<12}% | {it_acc:<18.2f}% | {os_acc:<18.2f}% | {r_mean:5.2f}% ± {r_std:<13.2f}")

# =====================================================================
# 4. Save Comparative Visualization Plot
# =====================================================================
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(retention_percentages, iterative_accs, 'o-', color='tab:green', linewidth=2.5, label='Iterative Causal Pruning (Recalculated)')
ax.plot(retention_percentages, oneshot_accs, 's--', color='tab:orange', linewidth=2, label='One-Shot Causal Pruning')
ax.errorbar(retention_percentages, random_acc_means, yerr=random_acc_stds, fmt='x-.', color='tab:blue', 
            linewidth=1.8, capsize=4, label=f'Layer-Wise Random (Mean ± Std, n={num_random_trials})')

ax.set_xlabel('Neuron Retention Percentage (%)', fontsize=12)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('Effect of Recalculating Importance: Iterative vs One-Shot Pruning', fontsize=14)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(fontsize=11)
plt.tight_layout()

plt.savefig('iterative_vs_oneshot_pruning.png', dpi=300)
plt.close()
print("\nPlot saved: 'iterative_vs_oneshot_pruning.png'")
