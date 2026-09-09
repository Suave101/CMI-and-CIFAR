import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import AgglomerativeClustering

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
# 2. Extract Behavioral Fingerprints & Compute Global Causal Scores
# =====================================================================
# Extract activation fingerprints (360-dim vector per neuron across reference set)
with torch.no_grad():
    h1_act = model.relu1(model.fc1(X_test_t)).cpu().numpy()  # (360, 256)
    h2_act = model.relu2(model.fc2(torch.tensor(h1_act))).cpu().numpy()  # (360, 128)

# Transpose so each row corresponds to a neuron's behavioral fingerprint
fingerprints_l1 = h1_act.T  # (256, 360)
fingerprints_l2 = h2_act.T  # (128, 360)
all_fingerprints = np.vstack([fingerprints_l1, fingerprints_l2])  # (384, 360)

# Pool exact causal ablation scores globally across all layers
scores_l1 = np.zeros(256)
scores_l2 = np.zeros(128)

with torch.no_grad():
    for i in range(256):
        h1 = model.relu1(model.fc1(X_test_t))
        h1[:, i] = 0.0
        h2 = model.relu2(model.fc2(h1))
        scores_l1[i] = (orig_test_preds != model.fc3(h2).argmax(dim=1)).float().mean().item()

    for j in range(128):
        h1 = model.relu1(model.fc1(X_test_t))
        h2 = model.relu2(model.fc2(h1))
        h2[:, j] = 0.0
        scores_l2[j] = (orig_test_preds != model.fc3(h2).argmax(dim=1)).float().mean().item()

all_scores = np.concatenate([scores_l1, scores_l2])  # Pooled across all layers

# =====================================================================
# 3. Causal Restructuring Methodology
# =====================================================================
def build_and_evaluate_abstract_network(k_clusters):
    """
    Builds abstract model by isolating Top 15% causal core, clustering the remaining
    85% into k groups with Agglomerative Clustering, selecting closest representatives,
    and merging weights.
    """
    total_neurons = len(all_scores)
    n_top = int(np.round(0.15 * total_neurons))  # Top 15% causal core (~58 neurons)
    
    ranked_indices = np.argsort(all_scores)
    top_causal_indices = set(ranked_indices[-n_top:])
    remaining_indices = np.array([idx for idx in range(total_neurons) if idx not in top_causal_indices])
    
    # Target reference fingerprint from Top 15% causal core
    core_fingerprint_mean = all_fingerprints[list(top_causal_indices)].mean(axis=0)

    # Agglomerative clustering on the remaining 85% neurons
    clustering = AgglomerativeClustering(n_clusters=k_clusters)
    cluster_labels = clustering.fit_predict(all_fingerprints[remaining_indices])

    # Copy baseline weights for merging
    W1 = model.fc1.weight.clone()
    b1 = model.fc1.bias.clone()
    W2 = model.fc2.weight.clone()
    b2 = model.fc2.bias.clone()
    W3 = model.fc3.weight.clone()
    b3 = model.fc3.bias.clone()

    kept_l1 = set(idx for idx in top_causal_indices if idx < 256)
    kept_l2 = set(idx - 256 for idx in top_causal_indices if idx >= 256)

    # Process each cluster: find representative and merge outgoing weights
    for c_id in range(k_clusters):
        cluster_neuron_indices = remaining_indices[cluster_labels == c_id]
        
        # Split cluster neurons by original layer
        l1_cluster = [idx for idx in cluster_neuron_indices if idx < 256]
        l2_cluster = [idx - 256 for idx in cluster_neuron_indices if idx >= 256]

        # Process Layer 1 cluster members
        if l1_cluster:
            dists = [np.linalg.norm(all_fingerprints[idx] - core_fingerprint_mean) for idx in l1_cluster]
            rep_l1 = l1_cluster[np.argmin(dists)]
            kept_l1.add(rep_l1)
            # Merge outgoing weights of discarded neurons in cluster onto representative
            for idx in l1_cluster:
                if idx != rep_l1:
                    W2[:, rep_l1] += W2[:, idx]

        # Process Layer 2 cluster members
        if l2_cluster:
            dists = [np.linalg.norm(all_fingerprints[idx + 256] - core_fingerprint_mean) for idx in l2_cluster]
            rep_l2 = l2_cluster[np.argmin(dists)]
            kept_l2.add(rep_l2)
            # Merge outgoing weights of discarded neurons in cluster onto representative
            for idx in l2_cluster:
                if idx != rep_l2:
                    W3[:, rep_l2] += W3[:, idx]

    # Mask and extract active sub-matrices
    mask_l1 = np.array([i in kept_l1 for i in range(256)])
    mask_l2 = np.array([j in kept_l2 for j in range(128)])

    with torch.no_grad():
        W1_sub = W1[mask_l1, :]
        b1_sub = b1[mask_l1]
        W2_sub = W2[mask_l2, :][:, mask_l1]
        b2_sub = b2[mask_l2]
        W3_sub = W3[:, mask_l2]
        b3_sub = b3

        h1_p = torch.relu(torch.matmul(X_test_t, W1_sub.T) + b1_sub)
        h2_p = torch.relu(torch.matmul(h1_p, W2_sub.T) + b2_sub)
        p_logits = torch.matmul(h2_p, W3_sub.T) + b3_sub
        p_preds = p_logits.argmax(dim=1)

        acc = (p_preds == y_test_t).float().mean().item() * 100
        agree = (p_preds == orig_test_preds).float().mean().item() * 100

    total_kept = len(kept_l1) + len(kept_l2)
    return acc, agree, total_kept

# =====================================================================
# 4. Run Experiment across Cluster Sizes (k)
# =====================================================================
k_values = [2, 5, 10, 20, 30, 40, 50, 75, 100]
abstract_accs, abstract_agrees, kept_counts = [], [], []

print("=== CAUSAL RESTRUCTURING ABSTRACT MODEL EVALUATION ===")
print(f"{'Clusters (k)':<12} | {'Depth (k+1)':<12} | {'Kept Neurons':<14} | {'Accuracy %':<12} | {'Agreement %':<12}")
print("-" * 72)

for k in k_values:
    acc, agree, total_kept = build_and_evaluate_abstract_network(k_clusters=k)
    abstract_accs.append(acc)
    abstract_agrees.append(agree)
    kept_counts.append(total_kept)
    print(f"{k:<12} | {k+1:<12} | {total_kept:<14} | {acc:<12.2f}% | {agree:<12.2f}%")

# =====================================================================
# 5. Plot Agreement & Accuracy Results
# =====================================================================
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('Number of Clusters (k) [Abstract Depth = k + 1]', fontsize=12)
ax1.set_ylabel('Agreement with Original Model (%)', color=color, fontsize=12)
ax1.plot(k_values, abstract_agrees, 'o-', color=color, linewidth=2.5, label='Agreement (%)')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.5)

ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('Test Accuracy (%)', color=color, fontsize=12)
ax2.plot(k_values, abstract_accs, 's--', color=color, linewidth=2, label='Accuracy (%)')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Causal Restructuring: Model Agreement & Accuracy vs Cluster Count (k)', fontsize=14)
fig.tight_layout()

plt.savefig('causal_restructuring_results.png', dpi=300)
plt.close()
print("\nPlot saved: 'causal_restructuring_results.png'")
