import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =====================================================================
# 1. Load Data and Baseline Model
# =====================================================================
data = torch.load('dataset.pt')
X_test_t, y_test_t = data['X_test'], data['y_test']

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        # Layer 1 - 64 -> 256
        self.fc1 = nn.Linear(64, 256)
        self.relu1 = nn.ReLU()

        # Layer 2 - 256 -> 128
        self.fc2 = nn.Linear(256, 128)

        # Layer 3 - 128 -> 10
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
    orig_logits = model(X_test_t)
    orig_test_preds = orig_logits.argmax(dim=1)

TOTAL_HIDDEN_NEURONS = 256 + 128  # 384 total hidden neurons

# =====================================================================
# 2. Extract Fingerprints & Causal Scores
# =====================================================================
with torch.no_grad():
    # Calculate the activations of the 256 neurons in layer 1
    h1_act = model.relu1(model.fc1(X_test_t)).cpu().numpy()

    # Calculate the activations of the 128 neurons in layer 2
    h2_act = model.relu2(model.fc2(torch.tensor(h1_act))).cpu().numpy()

# Transpose the activations
fp_l1 = h1_act.T  # (256, 360)
fp_l2 = h2_act.T  # (128, 360)

# Get the mean of the activations from each layer
mean_act_l1 = fp_l1.mean(axis=1)
mean_act_l2 = fp_l2.mean(axis=1)

# Set the activations that are zero to epsilon which is a
# very small number that is insignificant. This is so
# we can take the cosine similarity score.
fp_l1_safe = fp_l1.copy()
fp_l1_safe[np.all(fp_l1 == 0, axis=1)] = 1e-8

fp_l2_safe = fp_l2.copy()
fp_l2_safe[np.all(fp_l2 == 0, axis=1)] = 1e-8

# Create empty arrays for the causual (importance) scores of the two layers
scores_l1 = np.zeros(256)
scores_l2 = np.zeros(128)

# Calculate importance scores
with torch.no_grad():
    # Layer 1 ablation
    for i in range(256):
        h1 = model.relu1(model.fc1(X_test_t))
        h1[:, i] = 0.0  # Zero out target neuron
        h2 = model.relu2(model.fc2(h1))
        new_logits = model.fc3(h2)
        
        # Measure directional shift: higher score = greater deviation from original logits
        cos_sim = F.cosine_similarity(orig_logits, new_logits, dim=1)
        scores_l1[i] = (1.0 - cos_sim).mean().item()

    # Layer 2 ablation
    for j in range(128):
        h1 = model.relu1(model.fc1(X_test_t))
        h2 = model.relu2(model.fc2(h1))
        h2[:, j] = 0.0  # Zero out target neuron
        new_logits = model.fc3(h2)
        
        # Measure directional shift: higher score = greater deviation from original logits
        cos_sim = F.cosine_similarity(orig_logits, new_logits, dim=1)
        scores_l2[j] = (1.0 - cos_sim).mean().item()

def get_cosine_distance(u, v):
    norm_u, norm_v = np.linalg.norm(u), np.linalg.norm(v)
    if norm_u == 0.0 or norm_v == 0.0:
        return 1.0
    return 1.0 - (np.dot(u, v) / (norm_u * norm_v))

def find_cluster_medoid(fingerprints, members):
    # Return the only member directly when the cluster contains one neuron
    if len(members) == 1:
        return members[0]

    # Calculate the cluster center and select the member closest to it
    sub_fps = fingerprints[members]
    centroid = sub_fps.mean(axis=0)
    dists = [get_cosine_distance(fp, centroid) for fp in sub_fps]
    return members[np.argmin(dists)]

# =====================================================================
# 3. Method 1: Causal Restructuring Pipeline
# =====================================================================
def build_causal_abstract_network(k_clusters_per_layer):
    # Select the top 15% of neurons from each layer based on causal importance
    n_top_l1 = int(np.round(0.15 * 256))
    n_top_l2 = int(np.round(0.15 * 128))

    # Store the indices of the most causally important neurons
    top_l1 = set(np.argsort(scores_l1)[-n_top_l1:])
    top_l2 = set(np.argsort(scores_l2)[-n_top_l2:])

    # Store the remaining neurons that will be clustered and merged
    rem_l1 = np.array([i for i in range(256) if i not in top_l1])
    rem_l2 = np.array([j for j in range(128) if j not in top_l2])

    # Cluster the remaining layer 1 neurons by activation similarity
    c_l1 = AgglomerativeClustering(
        n_clusters=k_clusters_per_layer,
        metric='cosine',
        linkage='average'
    )
    labels_l1 = c_l1.fit_predict(fp_l1_safe[rem_l1])

    # Cluster the remaining layer 2 neurons by activation similarity
    c_l2 = AgglomerativeClustering(
        n_clusters=k_clusters_per_layer,
        metric='cosine',
        linkage='average'
    )
    labels_l2 = c_l2.fit_predict(fp_l2_safe[rem_l2])

    # Copy the model weights so the original model is not modified
    W1 = model.fc1.weight.detach().clone()
    b1 = model.fc1.bias.detach().clone()
    W2 = model.fc2.weight.detach().clone()
    b2 = model.fc2.bias.detach().clone()
    W3 = model.fc3.weight.detach().clone()
    b3 = model.fc3.bias.detach().clone()

    # Always keep the most causally important neurons
    kept_l1, kept_l2 = set(top_l1), set(top_l2)

    # Restructure Layer 1
    for cid in range(k_clusters_per_layer):
        # Get the original neuron indices belonging to this cluster
        members = rem_l1[labels_l1 == cid]

        # Select the neuron whose fingerprint is closest to the cluster center
        rep = find_cluster_medoid(fp_l1_safe, members)
        kept_l1.add(rep)

        # Keep neurons whose activation fingerprints are positively aligned
        aligned = [
            m for m in members
            if (1.0 - get_cosine_distance(fp_l1_safe[m], fp_l1_safe[rep])) > 0.0
        ] or [rep]

        # Calculate the total activation used to weight the merged neurons
        total_act_aligned = sum(mean_act_l1[m] for m in aligned) + 1e-8

        # Combine the incoming weights and biases of aligned neurons
        W1_combo = sum(
            ((mean_act_l1[m] + 1e-8) / total_act_aligned) * W1[m]
            for m in aligned
        )
        b1_combo = sum(
            ((mean_act_l1[m] + 1e-8) / total_act_aligned) * b1[m]
            for m in aligned
        )

        # Assign the combined weights and bias to the representative neuron
        W1[rep], b1[rep] = W1_combo, b1_combo

        # Use the representative activation to scale outgoing connections
        rep_act = mean_act_l1[rep] + 1e-8
        for m in members:
            if m != rep:
                # Redirect each removed neuron's outgoing weights to the representative
                W2[:, rep] += (mean_act_l1[m] / rep_act) * W2[:, m]

    # Restructure Layer 2
    for cid in range(k_clusters_per_layer):
        # Get the original neuron indices belonging to this cluster
        members = rem_l2[labels_l2 == cid]

        # Select the neuron whose fingerprint is closest to the cluster center
        rep = find_cluster_medoid(fp_l2_safe, members)
        kept_l2.add(rep)

        # Keep neurons whose activation fingerprints are positively aligned
        aligned = [
            m for m in members
            if (1.0 - get_cosine_distance(fp_l2_safe[m], fp_l2_safe[rep])) > 0.0
        ] or [rep]

        # Calculate the total activation used to weight the merged neurons
        total_act_aligned = sum(mean_act_l2[m] for m in aligned) + 1e-8

        # Combine the incoming weights and biases of aligned neurons
        W2_combo = sum(
            ((mean_act_l2[m] + 1e-8) / total_act_aligned) * W2[m]
            for m in aligned
        )
        b2_combo = sum(
            ((mean_act_l2[m] + 1e-8) / total_act_aligned) * b2[m]
            for m in aligned
        )

        # Assign the combined weights and bias to the representative neuron
        W2[rep], b2[rep] = W2_combo, b2_combo

        # Use the representative activation to scale outgoing connections
        rep_act = mean_act_l2[rep] + 1e-8
        for m in members:
            if m != rep:
                # Redirect each removed neuron's outgoing weights to the representative
                W3[:, rep] += (mean_act_l2[m] / rep_act) * W3[:, m]

    # Create masks identifying the neurons retained in each layer
    mask_l1 = np.array([i in kept_l1 for i in range(256)])
    mask_l2 = np.array([j in kept_l2 for j in range(128)])

    with torch.no_grad():
        # Recalculate layer 1 using only the retained neurons
        h1_p = torch.relu(
            torch.matmul(X_test_t, W1[mask_l1, :].T) + b1[mask_l1]
        )

        # Recalculate layer 2 using retained layer 1 and layer 2 neurons
        h2_p = torch.relu(
            torch.matmul(
                h1_p,
                W2[mask_l2, :][:, mask_l1].T
            ) + b2[mask_l2]
        )

        # Calculate output logits using the retained layer 2 neurons
        p_logits = torch.matmul(h2_p, W3[:, mask_l2].T) + b3
        p_preds = p_logits.argmax(dim=1)

        # Calculate accuracy relative to the true test labels
        acc = (p_preds == y_test_t).float().mean().item() * 100

        # Calculate agreement with the original model's predictions
        agree = (p_preds == orig_test_preds).float().mean().item() * 100

    # Return accuracy, agreement, and the number of retained neurons
    return acc, agree, len(kept_l1), len(kept_l2)


# =====================================================================
# 4. Method 2: Random Neuron Ablation Baseline
# =====================================================================
def eval_random_ablation(n_kept_l1, n_kept_l2, n_trials=10, seed=42):
    """
    Randomly retains n_kept_l1 and n_kept_l2 neurons across n_trials,
    ablating the rest without weight merging.
    """
    np.random.seed(seed)
    trial_accs, trial_agrees = [], []

    W1_orig = model.fc1.weight.detach()
    b1_orig = model.fc1.bias.detach()
    W2_orig = model.fc2.weight.detach()
    b2_orig = model.fc2.bias.detach()
    W3_orig = model.fc3.weight.detach()
    b3_orig = model.fc3.bias.detach()

    for _ in range(n_trials):
        kept_l1 = np.random.choice(256, n_kept_l1, replace=False)
        kept_l2 = np.random.choice(128, n_kept_l2, replace=False)

        W1_sub = W1_orig[kept_l1, :]
        b1_sub = b1_orig[kept_l1]
        W2_sub = W2_orig[kept_l2, :][:, kept_l1]
        b2_sub = b2_orig[kept_l2]
        W3_sub = W3_orig[:, kept_l2]

        with torch.no_grad():
            h1 = torch.relu(torch.matmul(X_test_t, W1_sub.T) + b1_sub)
            h2 = torch.relu(torch.matmul(h1, W2_sub.T) + b2_sub)
            p_logits = torch.matmul(h2, W3_sub.T) + b3_orig
            p_preds = p_logits.argmax(dim=1)

            acc = (p_preds == y_test_t).float().mean().item() * 100
            agree = (p_preds == orig_test_preds).float().mean().item() * 100

            trial_accs.append(acc)
            trial_agrees.append(agree)

    return np.mean(trial_accs), np.mean(trial_agrees)

# =====================================================================
# 5. Comparative Evaluation Across % Neurons Kept
# =====================================================================
k_values = [2, 5, 10, 20, 30, 40, 50, 75]

pct_kept_list = []
causal_accs, causal_agrees = [], []
rand_abl_accs, rand_abl_agrees = [], []

print("=== % NEURONS KEPT: OUR METHOD VS. RANDOM ABLATION ===")
print(f"{'% Kept':<10} | {'Kept Neurons':<12} | {'Causal Acc %':<13} | {'Rand Abl Acc %':<15} | {'Causal Agr %':<13} | {'Rand Abl Agr %':<15}")
print("-" * 86)

for k in k_values:
    c_acc, c_agr, kept_l1, kept_l2 = build_causal_abstract_network(k_clusters_per_layer=k)
    total_kept = kept_l1 + kept_l2
    pct_kept = (total_kept / TOTAL_HIDDEN_NEURONS) * 100.0

    r_acc, r_agr = eval_random_ablation(n_kept_l1=kept_l1, n_kept_l2=kept_l2, n_trials=10)

    pct_kept_list.append(pct_kept)
    causal_accs.append(c_acc)
    causal_agrees.append(c_agr)
    rand_abl_accs.append(r_acc)
    rand_abl_agrees.append(r_agr)

    print(f"{pct_kept:<10.2f}% | {total_kept:<12} | {c_acc:<13.2f} | {r_acc:<15.2f} | {c_agr:<13.2f} | {r_agr:<15.2f}")

# =====================================================================
# 6. Plotting Results (% Neurons Kept on X-Axis)
# =====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Agreement vs % Neurons Kept
ax1.plot(pct_kept_list, causal_agrees, 'o-', color='tab:blue', linewidth=2.5, label='Our Causal Method')
ax1.plot(pct_kept_list, rand_abl_agrees, 'o--', color='tab:red', linewidth=2.0, label='Random Neuron Ablation')
ax1.set_xlabel('Percentage of Neurons Kept (%)', fontsize=12)
ax1.set_ylabel('Agreement with Original Model (%)', fontsize=12)
ax1.set_title('Agreement vs. % Neurons Kept', fontsize=14)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(fontsize=11)

# Plot 2: Accuracy vs % Neurons Kept
ax2.plot(pct_kept_list, causal_accs, 's-', color='tab:green', linewidth=2.5, label='Our Causal Method')
ax2.plot(pct_kept_list, rand_abl_accs, 's--', color='tab:orange', linewidth=2.0, label='Random Neuron Ablation')
ax2.set_xlabel('Percentage of Neurons Kept (%)', fontsize=12)
ax2.set_ylabel('Test Accuracy (%)', fontsize=12)
ax2.set_title('Test Accuracy vs. % Neurons Kept', fontsize=14)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(fontsize=11)

plt.tight_layout()
plt.savefig('pct_kept_causal_vs_random_ablation.png', dpi=300)
plt.savefig('pct_kept_causal_vs_random_ablation.svg')
plt.close()