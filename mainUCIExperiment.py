import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import torch.nn.functional as F

# =====================================================================
# 0. Configuration & Reproducibility
# =====================================================================
SEEDS = [42, 101, 2024, 7, 99, 421, 1234, 5678, 91011, 121314, 8675309, 31415, 271828, 1618033, 3141592]
TOTAL_HIDDEN_NEURONS = 384
target_budgets = [38, 60, 73, 85, 108, 135, 160, 200, 225]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =====================================================================
# 1. Model & Data Definitions
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
        return self.fc3(h2)

class EMAModel:
    def __init__(self, model, decay=0.98):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.data.clone() for name, param in self.model.named_parameters() if param.requires_grad}
        self.backup = {}

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])

class AbstractSubMLP(nn.Module):
    def __init__(self, w1, b1, w2, b2, w3, b3):
        super().__init__()
        self.fc1 = nn.Linear(w1.shape[1], w1.shape[0])
        self.fc2 = nn.Linear(w2.shape[1], w2.shape[0])
        self.fc3 = nn.Linear(w3.shape[1], w3.shape[0])

        self.fc1.weight = nn.Parameter(w1.clone())
        self.fc1.bias = nn.Parameter(b1.clone())
        self.fc2.weight = nn.Parameter(w2.clone())
        self.fc2.bias = nn.Parameter(b2.clone())
        self.fc3.weight = nn.Parameter(w3.clone())
        self.fc3.bias = nn.Parameter(b3.clone())

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def kd_loss_fn(student_logits, teacher_logits, labels, T=2.0, alpha=0.85):
    soft_targets = F.softmax(teacher_logits / T, dim=1)
    soft_prob = F.log_softmax(student_logits / T, dim=1)
    kl_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (T ** 2)
    ce_loss = F.cross_entropy(student_logits, labels)
    return alpha * kl_loss + (1.0 - alpha) * ce_loss

def find_active_cluster_medoid(fingerprints, mean_acts, members):
    if len(members) == 1:
        return members[0]
    sub_fps = fingerprints[members]
    centroid = sub_fps.mean(axis=0)
    dists = np.array([1.0 - np.dot(fp, centroid)/(np.linalg.norm(fp)*np.linalg.norm(centroid)+1e-8) for fp in sub_fps])
    act_penalty = np.where(mean_acts[members] < 1e-5, 10.0, 0.0)
    return members[np.argmin(dists + act_penalty)]

def get_optimal_reserve_ratio(target_budget_pct):
    if target_budget_pct <= 12.0:
        return 0.00
    elif target_budget_pct <= 35.0:
        return 0.08
    else:
        return 0.15

# =====================================================================
# 2. Benchmark Runner across Seeds
# =====================================================================
def run_single_seed_benchmark(seed):
    set_seed(seed)
    data = torch.load('dataset.pt')
    X_raw, y_raw = data['X_train'].to(device), data['y_train'].to(device)
    X_test_t, y_test_t = data['X_test'].to(device), data['y_test'].to(device)

    val_size = int(0.2 * len(X_raw))
    train_size = len(X_raw) - val_size
    
    # Random permutation per seed for Train/Val split
    perm = torch.randperm(len(X_raw))
    X_train_t, X_val_t = X_raw[perm[:train_size]], X_raw[perm[train_size:]]
    y_train_t, y_val_t = y_raw[perm[:train_size]], y_raw[perm[train_size:]]

    model = MLP().to(device)
    model.load_state_dict(torch.load('mlp_model.pth', map_location=device))
    model.eval()

    with torch.no_grad():
        orig_test_logits = model(X_test_t)
        orig_test_preds = orig_test_logits.argmax(dim=1)
        train_teacher_logits = model(X_train_t)
        val_teacher_logits = model(X_val_t)

    # Profiling on Validation set ONLY (No Leakage)
    with torch.no_grad():
        h1_act = model.relu1(model.fc1(X_val_t)).cpu().numpy()
        h2_act = model.relu2(model.fc2(torch.tensor(h1_act, device=device))).cpu().numpy()

    fp_l1, fp_l2 = h1_act.T, h2_act.T
    mean_act_l1, mean_act_l2 = fp_l1.mean(axis=1), fp_l2.mean(axis=1)

    fp_l1_safe, fp_l2_safe = fp_l1.copy(), fp_l2.copy()
    fp_l1_safe[np.all(fp_l1 == 0, axis=1)] = 1e-8
    fp_l2_safe[np.all(fp_l2 == 0, axis=1)] = 1e-8

    scores_l1, scores_l2 = np.zeros(256), np.zeros(128)
    with torch.no_grad():
        for i in range(256):
            h1 = model.relu1(model.fc1(X_val_t))
            h1[:, i] = 0.0
            scores_l1[i] = (1.0 - F.cosine_similarity(val_teacher_logits, model.fc3(model.relu2(model.fc2(h1))), dim=1)).mean().item()

        for j in range(128):
            h1 = model.relu1(model.fc1(X_val_t))
            h2 = model.relu2(model.fc2(h1))
            h2[:, j] = 0.0
            scores_l2[j] = (1.0 - F.cosine_similarity(val_teacher_logits, model.fc3(h2), dim=1)).mean().item()

    def fine_tune(sub_w1, sub_b1, sub_w2, sub_b2, sub_w3, sub_b3):
        sub_model = AbstractSubMLP(sub_w1, sub_b1, sub_w2, sub_b2, sub_w3, sub_b3).to(device)
        optimizer = optim.Adam(sub_model.parameters(), lr=1e-3)
        ema = EMAModel(sub_model, decay=0.98)
        dataset = TensorDataset(X_train_t, y_train_t, train_teacher_logits)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        sub_model.train()
        for _ in range(5):
            for bx, by, b_tlogits in loader:
                optimizer.zero_grad()
                s_logits = sub_model(bx)
                loss = kd_loss_fn(s_logits, b_tlogits, by)
                loss.backward()
                optimizer.step()
                ema.update()

        sub_model.eval()
        with torch.no_grad():
            preds_no_ema = sub_model(X_test_t).argmax(dim=1)
            acc_no_ema = (preds_no_ema == y_test_t).float().mean().item() * 100.0
            agr_no_ema = (preds_no_ema == orig_test_preds).float().mean().item() * 100.0

        ema.apply_shadow()
        with torch.no_grad():
            preds_ema = sub_model(X_test_t).argmax(dim=1)
            acc_ema = (preds_ema == y_test_t).float().mean().item() * 100.0
            agr_ema = (preds_ema == orig_test_preds).float().mean().item() * 100.0
        ema.restore()
        return acc_no_ema, agr_no_ema, acc_ema, agr_ema

    def build_core(target_budget_neurons, use_causal=True):
        t0_zs = time.time()
        if use_causal:
            pct_target = (target_budget_neurons / TOTAL_HIDDEN_NEURONS) * 100.0
            reserve_ratio = get_optimal_reserve_ratio(pct_target)
            budget_l1 = max(1, int(np.round(target_budget_neurons * (256 / 384))))
            budget_l2 = max(1, target_budget_neurons - budget_l1)
            n_top_l1 = int(np.round(budget_l1 * reserve_ratio))
            n_top_l2 = int(np.round(budget_l2 * reserve_ratio))
            k_c1, k_c2 = max(1, budget_l1 - n_top_l1), max(1, budget_l2 - n_top_l2)
            top_l1 = set(np.argsort(scores_l1)[-n_top_l1:]) if n_top_l1 > 0 else set()
            top_l2 = set(np.argsort(scores_l2)[-n_top_l2:]) if n_top_l2 > 0 else set()
        else:
            k_c1, k_c2 = max(1, target_budget_neurons[0]), max(1, target_budget_neurons[1])
            top_l1, top_l2 = set(), set()

        rem_l1 = np.array([i for i in range(256) if i not in top_l1])
        rem_l2 = np.array([j for j in range(128) if j not in top_l2])

        W1, b1 = model.fc1.weight.detach().clone(), model.fc1.bias.detach().clone()
        W2, b2 = model.fc2.weight.detach().clone(), model.fc2.bias.detach().clone()
        W3, b3 = model.fc3.weight.detach().clone(), model.fc3.bias.detach().clone()

        kept_l1, kept_l2 = set(top_l1), set(top_l2)

        if len(rem_l1) > 0 and k_c1 > 0:
            c1 = AgglomerativeClustering(n_clusters=min(k_c1, len(rem_l1)), metric='cosine', linkage='average')
            labels_l1 = c1.fit_predict(fp_l1_safe[rem_l1])
            for cid in range(k_c1):
                m = rem_l1[labels_l1 == cid]
                if len(m) == 0: continue
                rep = find_active_cluster_medoid(fp_l1_safe, mean_act_l1, m)
                kept_l1.add(rep)
                W1[rep], b1[rep] = W1[m].mean(dim=0), b1[m].mean(dim=0)
                norm = np.sqrt(len(m))
                for x in m:
                    if x != rep: W2[:, rep] += W2[:, x] / norm

        if len(rem_l2) > 0 and k_c2 > 0:
            c2 = AgglomerativeClustering(n_clusters=min(k_c2, len(rem_l2)), metric='cosine', linkage='average')
            labels_l2 = c2.fit_predict(fp_l2_safe[rem_l2])
            for cid in range(k_c2):
                m = rem_l2[labels_l2 == cid]
                if len(m) == 0: continue
                rep = find_active_cluster_medoid(fp_l2_safe, mean_act_l2, m)
                kept_l2.add(rep)
                W2[rep], b2[rep] = W2[m].mean(dim=0), b2[m].mean(dim=0)
                norm = np.sqrt(len(m))
                for x in m:
                    if x != rep: W3[:, rep] += W3[:, x] / norm

        m_l1 = np.array([i in kept_l1 for i in range(256)])
        m_l2 = np.array([j in kept_l2 for j in range(128)])
        sub_w1, sub_b1 = W1[m_l1, :], b1[m_l1]
        sub_w2, sub_b2 = W2[m_l2, :][:, m_l1], b2[m_l2]
        sub_w3, sub_b3 = W3[:, m_l2], b3

        with torch.no_grad():
            h1_p = torch.relu(torch.matmul(X_test_t, sub_w1.T) + sub_b1)
            h2_p = torch.relu(torch.matmul(h1_p, sub_w2.T) + sub_b2)
            preds = (torch.matmul(h2_p, sub_w3.T) + sub_b3).argmax(dim=1)
            zs_acc = (preds == y_test_t).float().mean().item() * 100.0
            zs_agr = (preds == orig_test_preds).float().mean().item() * 100.0

        t_zs = (time.time() - t0_zs) * 1000.0
        t0_ft = time.time()
        no_ema_acc, no_ema_agr, ema_acc, ema_agr = fine_tune(sub_w1, sub_b1, sub_w2, sub_b2, sub_w3, sub_b3)
        t_ft = (time.time() - t0_ft) * 1000.0

        return zs_acc, zs_agr, no_ema_acc, no_ema_agr, ema_acc, ema_agr, len(kept_l1), len(kept_l2), t_zs, t_ft

    seed_res = {
        'our_zs_acc': [], 'our_zs_agr': [], 'our_noema_acc': [], 'our_noema_agr': [], 'our_ema_acc': [], 'our_ema_agr': [],
        'da_zs_acc': [], 'da_zs_agr': [], 'da_ft_acc': [], 'da_ft_agr': [],
        'pct_kept': [], 'our_zs_time': [], 'our_ft_time': [], 'da_zs_time': [], 'da_ft_time': []
    }

    for b in target_budgets:
        o_zs_acc, o_zs_agr, o_ne_acc, o_ne_agr, o_e_acc, o_e_agr, n1, n2, o_tzs, o_tft = build_core(b, use_causal=True)
        d_zs_acc, d_zs_agr, _, _, d_e_acc, d_e_agr, _, _, d_tzs, d_tft = build_core((n1, n2), use_causal=False)

        pct = ((n1 + n2) / TOTAL_HIDDEN_NEURONS) * 100.0
        seed_res['pct_kept'].append(pct)
        seed_res['our_zs_acc'].append(o_zs_acc); seed_res['our_zs_agr'].append(o_zs_agr)
        seed_res['our_noema_acc'].append(o_ne_acc); seed_res['our_noema_agr'].append(o_ne_agr)
        seed_res['our_ema_acc'].append(o_e_acc); seed_res['our_ema_agr'].append(o_e_agr)
        seed_res['da_zs_acc'].append(d_zs_acc); seed_res['da_zs_agr'].append(d_zs_agr)
        seed_res['da_ft_acc'].append(d_e_acc); seed_res['da_ft_agr'].append(d_e_agr)
        seed_res['our_zs_time'].append(o_tzs); seed_res['our_ft_time'].append(o_tft)
        seed_res['da_zs_time'].append(d_tzs); seed_res['da_ft_time'].append(d_tft)

    return seed_res

# Execute across all seeds
print(f"=== RUNNING MULTI-SEED BENCHMARK (N={len(SEEDS)}) ===")
all_runs = [run_single_seed_benchmark(s) for s in SEEDS]

pct_axis = np.mean([r['pct_kept'] for r in all_runs], axis=0)

def aggregate_metric(key):
    data = np.array([r[key] for r in all_runs])
    return np.mean(data, axis=0), np.std(data, axis=0)

our_ema_acc_m, our_ema_acc_s = aggregate_metric('our_ema_acc')
our_ema_agr_m, our_ema_agr_s = aggregate_metric('our_ema_agr')
our_noema_acc_m, our_noema_acc_s = aggregate_metric('our_noema_acc')
our_noema_agr_m, our_noema_agr_s = aggregate_metric('our_noema_agr')
our_zs_acc_m, our_zs_acc_s = aggregate_metric('our_zs_acc')
our_zs_agr_m, our_zs_agr_s = aggregate_metric('our_zs_agr')

da_ft_acc_m, da_ft_acc_s = aggregate_metric('da_ft_acc')
da_ft_agr_m, da_ft_agr_s = aggregate_metric('da_ft_agr')
da_zs_acc_m, da_zs_acc_s = aggregate_metric('da_zs_acc')
da_zs_agr_m, da_zs_agr_s = aggregate_metric('da_zs_agr')

our_zs_t_m, our_zs_t_s = aggregate_metric('our_zs_time')
our_ft_t_m, our_ft_t_s = aggregate_metric('our_ft_time')
da_zs_t_m, da_zs_t_s = aggregate_metric('da_zs_time')
da_ft_t_m, da_ft_t_s = aggregate_metric('da_ft_time')

# Print Summary Table
print("\n" + "="*80)
print(f"{'Budget (%)':<12} | {'Our FT (EMA) Acc (%)':<24} | {'DA FT Acc (%)':<22} | {'Delta (%)':<10}")
print("="*80)
for i in range(len(pct_axis)):
    print(f"{pct_axis[i]:<12.1f} | {our_ema_acc_m[i]:.2f} ± {our_ema_acc_s[i]:.2f}{'':<12} | {da_ft_acc_m[i]:.2f} ± {da_ft_acc_s[i]:.2f}{'':<10} | {our_ema_acc_m[i] - da_ft_acc_m[i]:+.2f}")
print("="*80)

# =====================================================================
# 3. Plotting Figure with Multi-Seed Variance Bands
# =====================================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6))

def plot_with_band(ax, x, mean, std, fmt, color, label):
    ax.plot(x, mean, fmt, color=color, linewidth=2.0, label=label)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

# Panel 1: Agreement
plot_with_band(ax1, pct_axis, our_ema_agr_m, our_ema_agr_s, 'D-', '#008080', 'Our Method (KD FT + EMA)')
plot_with_band(ax1, pct_axis, our_noema_agr_m, our_noema_agr_s, 'o--', '#20B2AA', 'Our Method (KD FT - No EMA)')
plot_with_band(ax1, pct_axis, our_zs_agr_m, our_zs_agr_s, 's-.', '#1f77b4', 'Our Method (Zero-Shot)')
plot_with_band(ax1, pct_axis, da_ft_agr_m, da_ft_agr_s, '^-', '#6a3d9a', 'DeepAbstract (KD FT)')
plot_with_band(ax1, pct_axis, da_zs_agr_m, da_zs_agr_s, 'v-.', '#9467bd', 'DeepAbstract (Zero-Shot)')

ax1.set_xlabel('Percentage of Neurons Kept (%)', fontsize=11)
ax1.set_ylabel('Agreement with Original Model (%)', fontsize=11)
ax1.set_title('Agreement vs. % Neurons Kept (Mean ± Std)', fontsize=13)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(fontsize=8, loc='lower right')

# Panel 2: Test Accuracy
plot_with_band(ax2, pct_axis, our_ema_acc_m, our_ema_acc_s, 'D-', '#008080', 'Our Method (KD FT + EMA)')
plot_with_band(ax2, pct_axis, our_noema_acc_m, our_noema_acc_s, 'o--', '#20B2AA', 'Our Method (KD FT - No EMA)')
plot_with_band(ax2, pct_axis, our_zs_acc_m, our_zs_acc_s, 's-.', '#2ca02c', 'Our Method (Zero-Shot)')
plot_with_band(ax2, pct_axis, da_ft_acc_m, da_ft_acc_s, '^-', '#6a3d9a', 'DeepAbstract (KD FT)')
plot_with_band(ax2, pct_axis, da_zs_acc_m, da_zs_acc_s, 'v-.', '#9467bd', 'DeepAbstract (Zero-Shot)')

ax2.set_xlabel('Percentage of Neurons Kept (%)', fontsize=11)
ax2.set_ylabel('Test Accuracy (%)', fontsize=11)
ax2.set_title('Test Accuracy vs. % Neurons Kept (Mean ± Std)', fontsize=13)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(fontsize=8, loc='lower right')

# Panel 3: Execution Runtime
methods = ['DeepAbstract', 'Our Method']
x_idx = np.arange(len(methods))
bw = 0.35

mean_zs_t = [np.mean(da_zs_t_m), np.mean(our_zs_t_m)]
std_zs_t = [np.std(da_zs_t_m), np.std(our_zs_t_m)]
mean_ft_t = [np.mean(da_ft_t_m), np.mean(our_ft_t_m)]
std_ft_t = [np.std(da_ft_t_m), np.std(our_ft_t_m)]

r1 = ax3.bar(x_idx - bw/2, mean_zs_t, bw, yerr=std_zs_t, capsize=4, label='Zero-Shot Runtime', color='#3498db', edgecolor='black', linewidth=0.8)
r2 = ax3.bar(x_idx + bw/2, mean_ft_t, bw, yerr=std_ft_t, capsize=4, label='Fine-Tuned (KD + EMA) Runtime', color='#2ecc71', edgecolor='black', linewidth=0.8)

ax3.set_ylabel('Average Runtime per Run (ms)', fontsize=11)
ax3.set_title('Average Execution Runtime Comparison', fontsize=13)
ax3.set_xticks(x_idx)
ax3.set_xticklabels(methods, fontsize=10)
ax3.grid(True, axis='y', linestyle='--', alpha=0.5)
ax3.legend(fontsize=9, loc='upper left')

for rect in r1:
    h = rect.get_height()
    ax3.annotate(f'{h:.1f} ms', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

for rect in r2:
    h = rect.get_height()
    ax3.annotate(f'{h:.1f} ms', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('clean_benchmark_multiseed.png', dpi=300)
plt.close()

print("\nBenchmark complete. Multi-seed plot saved to 'clean_benchmark_multiseed.png'")