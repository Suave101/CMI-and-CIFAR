import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import matplotlib
matplotlib.use('Agg')  # Headless backend
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
# 2. Extract Behavioral Fingerprints & Compute Causal Ablation Scores
# =====================================================================
# Behavioral fingerprint: Vector of neuron activations across input set (Slide 2)
with torch.no_grad():
    h1_act = model.relu1(model.fc1(X_test_t)).cpu().numpy()  # [360, 256]
    h2_act = model.relu2(model.fc2(model.relu1(model.fc1(X_test_t)))).cpu().numpy()  # [360, 128]

# Stack fingerprints across all 384 neurons -> [384, 360]
all_fingerprints = np.vstack([h1_act.T, h2_act.T])

# Step 1: Score every neuron via individual ablation across all layers (Slide 3)
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

# Pool scores globally across layers
all_scores = np.concatenate([scores_l1, scores_l2])

# =====================================================================
# 3. Restructuring Algorithm (Top 15% L1 + Agglomerative k-Clusters)
# =====================================================================
num_neurons = len(all_scores)
top_15_k = int(round(0.15 * num_neurons))  # Top 15% (~58 neurons)

ranked_indices = np.argsort(all_scores)
top_15_indices = ranked_indices[-top_15_k:]  # Layer 1 neurons
rem_indices = ranked_indices[:-top_15_k]    # Remaining 85% neurons

# Centroid of top 15% behavioral fingerprints
top_causal_center = np.mean(all_fingerprints[top_15_indices], axis=0)

def evaluate_causal_restructure(k_clusters):
    """
    Builds and evaluates abstract network of depth k + 1 (Slide 3 & 5).
    """
    # Step 3: Cluster remaining 85% into k groups via Agglomerative Clustering (Slide 3)
    clustering = AgglomerativeClustering(n_clusters=k_clusters)
    cluster_labels = clustering.fit_predict(all_fingerprints[rem_indices])

    selected_cluster_neurons = []
    
    # Step 4: Keep neuron closest to top causal group per cluster (Slide 3)
    for c in range(k_clusters):
        c_mask = (cluster_labels == c)
        c_indices = rem_indices[c_mask]
        
        c_fingerprints = all_fingerprints[c_indices]
        distances = np.linalg.norm(c_fingerprints - top_causal_center, axis=1)
        
        best_neuron = c_indices[np.argmin(distances)]
        selected_cluster_neurons.append(best_neuron)

    # Topology: Layer 1 = Top 15%, Layers 2..(k+1) = Cluster Representatives
    abstract_layers = [top_15_indices] + [[idx] for idx in selected_cluster_neurons]
    
    with torch.no_grad():
        # --- Construct Abstract Layer 1 (Input 64 -> Top 15% Neurons) ---
        l1_idxs = abstract_layers[0]
        l1_w, l1_b = [], []
        for idx in l1_idxs:
            if idx < 256:  # Neuron from orig L1
                l1_w.append(model.fc1.weight[idx].numpy())
                l1_b.append(model.fc1.bias[idx].item())
            else:  # Neuron from orig L2 (project incoming L1 weights to input)
                l2_idx = idx - 256
                proj_w = model.fc2.weight[l2_idx].numpy() @ model.fc1.weight.numpy()
                l1_w.append(proj_w)
                l1_b.append(model.fc2.bias[l2_idx].item())
                
        W1 = torch.tensor(np.array(l1_w), dtype=torch.float32)
        b1 = torch.tensor(np.array(l1_b), dtype=torch.float32)
        
        curr_act = torch.relu(torch.matmul(X_test_t, W1.T) + b1)  # [360, 58]
        
        # --- Construct Sequential Cluster Layers (Layers 2 to k+1) ---
        prev_indices = l1_idxs
        for layer_idx in range(1, len(abstract_layers)):
            curr_neuron = abstract_layers[layer_idx][0]
            w_conn = []
            for p_idx in prev_indices:
                if p_idx < 256 and curr_neuron >= 256:
                    w_conn.append(model.fc2.weight[curr_neuron - 256, p_idx].item())
                elif p_idx == curr_neuron:
                    w_conn.append(1.0)
                else:
                    w_conn.append(0.0)
            
            b_val = model.fc1.bias[curr_neuron].item() if curr_neuron < 256 else model.fc2.bias[curr_neuron - 256].item()
            
            W_step = torch.tensor(np.array([w_conn]), dtype=torch.float32)  # [1, prev_dim]
            b_step = torch.tensor(np.array([b_val]), dtype=torch.float32)   # [1]
            
            curr_act = torch.relu(torch.matmul(curr_act, W_step.T) + b_step)  # [360, 1]
            prev_indices = abstract_layers[layer_idx]
            
        # --- Construct Output Layer (10 classes) ---
        last_neuron = prev_indices[0]
        if last_neuron >= 256:
            W_out = model.fc3.weight[:, last_neuron - 256]  # [10]
        else:
            W_out = torch.tensor(
                model.fc3.weight.numpy() @ model.fc2.weight.numpy()[:, last_neuron], 
                dtype=torch.float32
            )  # [10]
            
        b_out = model.fc3.bias  # [10]
        
        # Shape alignment: [360, 1] @ [1, 10] -> [360, 10]
        W_out_proj = W_out.unsqueeze(0)  # [1, 10]
        final_logits = torch.matmul(curr_act, W_out_proj) + b_out
        preds = final_logits.argmax(dim=1)
        
        acc = (preds == y_test_t).float().mean().item() * 100
        agree = (preds == orig_test_preds).float().mean().item() * 100
        
    return acc, agree

# =====================================================================
# 4. Run Evaluation Across Target Depths
# =====================================================================
k_values = [2, 3, 4, 5, 8, 10]
print("=== CAUSAL RESTRUCTURING METHODOLOGY EVALUATION ===")
print(f"{'Clusters (k)':<14} | {'Total Depth (k+1)':<18} | {'Accuracy %':<12} | {'Agreement %':<12}")
print("-" * 62)

accs, agrees = [], []
for k in k_values:
    acc, agree = evaluate_causal_restructure(k)
    accs.append(acc)
    agrees.append(agree)
    print(f"{k:<14} | {k+1:<18} | {acc:<12.2f} | {agree:<12.2f}")

# Save Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot([k + 1 for k in k_values], accs, 'o-', color='tab:green', label='Abstract Model Accuracy')
ax.plot([k + 1 for k in k_values], agrees, 's--', color='tab:blue', label='Abstract Model Agreement')
ax.set_xlabel('Abstract Model Depth (k + 1)')
ax.set_ylabel('Percentage (%)')
ax.set_title('Causal Restructuring: Depth vs Accuracy & Agreement')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()
plt.tight_layout()
plt.savefig('causal_restructuring_depth_eval.png', dpi=300)
plt.close()
