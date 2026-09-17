import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import torch.nn.functional as F

# =====================================================================
# 0. Configuration & Reproducibility
# =====================================================================
SEEDS = [42, 101, 2024, 7, 99]
TOTAL_HIDDEN_NEURONS = 256 + 128  # 384 Total Hidden Neurons
target_budgets = [38, 60, 73, 85, 108, 135, 160, 200, 225]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    """
    Generates deterministic results across runs by setting seeds for random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_safe_fp(fp):
    """
    Prevents zero-vector runtime issues in cosine metric calculations.
    """
    fp_safe = fp.copy()
    zero_mask = np.all(fp_safe == 0, axis=1)
    if np.any(zero_mask):
        fp_safe[zero_mask] = 1e-8
    return fp_safe


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
        self.shadow = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        self.backup = {}

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    1.0 - self.decay
                ) * param.data + self.decay * self.shadow[name]

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
    kl_loss = F.kl_div(soft_prob, soft_targets, reduction="batchmean") * (T**2)
    ce_loss = F.cross_entropy(student_logits, labels)
    return alpha * kl_loss + (1.0 - alpha) * ce_loss


def find_active_cluster_medoid(fingerprints, mean_acts, members):
    if len(members) == 1:
        return members[0]
    sub_fps = fingerprints[members]
    centroid = sub_fps.mean(axis=0)
    dists = np.array(
        [
            1.0
            - np.dot(fp, centroid)
            / (np.linalg.norm(fp) * np.linalg.norm(centroid) + 1e-8)
            for fp in sub_fps
        ]
    )
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
# 2. Global Causual Restructuring Module (Global Ablation, Clustering & Restructuring)
# =====================================================================
def run_ablation_clustering_restructure_workflow(
    teacher_model,
    X_val_t,
    val_teacher_logits,
    top_n_pct=15.0,
    n_clusters=4,
    prune_pct_per_layer=20.0,
    device=device,
):
    """
    Executes the 6-step workflow for the 3-FC layer MLP architecture:
    1. Single Neuron Ablation
    2. Behavioral Fingerprinting
    3. Keep top n% of neurons
    4. Agglomerative Clustering of (100-n)% of neurons
    5. Restructure network such that each cluster is a layer and layer 1 is top n%
    6. Remove least important neurons from each layer
    """
    teacher_model.eval()

    # 1. Behavioral Fingerprinting
    with torch.no_grad():
        h1_act = teacher_model.relu1(teacher_model.fc1(X_val_t)).cpu().numpy()
        h2_act = (
            teacher_model.relu2(teacher_model.fc2(torch.tensor(h1_act, device=device)))
            .cpu()
            .numpy()
        )

    fp_l1, fp_l2 = h1_act.T, h2_act.T
    all_fps = np.vstack([fp_l1, fp_l2])  # (384, N_val)
    all_fps_safe = make_safe_fp(all_fps)

    neuron_map = [("fc1", i) for i in range(256)] + [("fc2", j) for j in range(128)]
    ablation_scores = np.zeros(len(neuron_map))

    # Single Neuron Ablation Scoring
    with torch.no_grad():
        for i in range(256):
            h1 = teacher_model.relu1(teacher_model.fc1(X_val_t))
            h1[:, i] = 0.0
            out = teacher_model.fc3(teacher_model.relu2(teacher_model.fc2(h1)))
            ablation_scores[i] = (
                (1.0 - F.cosine_similarity(val_teacher_logits, out, dim=1))
                .mean()
                .item()
            )

        for j in range(128):
            h1 = teacher_model.relu1(teacher_model.fc1(X_val_t))
            h2 = teacher_model.relu2(teacher_model.fc2(h1))
            h2[:, j] = 0.0
            out = teacher_model.fc3(h2)
            ablation_scores[256 + j] = (
                (1.0 - F.cosine_similarity(val_teacher_logits, out, dim=1))
                .mean()
                .item()
            )

    # Step 3: Keep top n% neurons globally
    total_neurons = len(neuron_map)
    n_top = int(np.ceil((top_n_pct / 100.0) * total_neurons))
    sorted_indices = np.argsort(ablation_scores)[::-1]
    top_n_indices = sorted_indices[:n_top]
    rem_indices = sorted_indices[n_top:]

    # Step 4: Clustering remaining (100-n)% neurons
    rem_fps = all_fps_safe[rem_indices]
    actual_clusters = min(n_clusters, len(rem_indices))
    if actual_clusters > 1:
        clustering = AgglomerativeClustering(
            n_clusters=actual_clusters, metric="cosine", linkage="average"
        )
        cluster_labels = clustering.fit_predict(rem_fps)
    else:
        cluster_labels = np.zeros(len(rem_indices), dtype=int)

    # Step 5: Restructure layers
    restructured_layers = {1: top_n_indices.tolist()}
    for c_id in range(actual_clusters):
        cluster_members = rem_indices[cluster_labels == c_id].tolist()
        restructured_layers[c_id + 2] = cluster_members

    # Step 6: Prune least important neurons inside each layer
    pruned_layers = {}
    for layer_idx, neurons in restructured_layers.items():
        if len(neurons) == 0:
            pruned_layers[layer_idx] = []
            continue
        layer_scores = ablation_scores[neurons]
        sorted_in_layer = np.argsort(layer_scores)
        n_remove = int(np.floor(len(neurons) * (prune_pct_per_layer / 100.0)))
        kept_in_layer = [neurons[idx] for idx in sorted_in_layer[n_remove:]]
        pruned_layers[layer_idx] = kept_in_layer

    # Map pruned indices back to original 3 FC layer structure
    all_kept = []
    for layer_idx in sorted(pruned_layers.keys()):
        all_kept.extend(pruned_layers[layer_idx])

    kept1 = [i for i in all_kept if i < 256]
    kept2 = [i - 256 for i in all_kept if i >= 256]

    if len(kept1) == 0:
        kept1 = [0]
    if len(kept2) == 0:
        kept2 = [0]

    w1 = teacher_model.fc1.weight.detach()[kept1, :]
    b1 = teacher_model.fc1.bias.detach()[kept1]

    w2 = teacher_model.fc2.weight.detach()[kept2, :][:, kept1]
    b2 = teacher_model.fc2.bias.detach()[kept2]

    w3 = teacher_model.fc3.weight.detach()[:, kept2]
    b3 = teacher_model.fc3.bias.detach()

    restructured_net = AbstractSubMLP(w1, b1, w2, b2, w3, b3).to(device)

    return {
        "ablation_scores": ablation_scores,
        "top_n_indices": top_n_indices,
        "restructured_layers": restructured_layers,
        "pruned_layers": pruned_layers,
        "restructured_model": restructured_net,
    }


# =====================================================================
# 3. Benchmark Runner across Seeds
# =====================================================================
def run_single_seed_benchmark(seed):
    set_seed(seed)
    data = torch.load("dataset.pt")
    X_raw, y_raw = data["X_train"].to(device), data["y_train"].to(device)
    X_test_t, y_test_t = data["X_test"].to(device), data["y_test"].to(device)

    val_size = int(0.2 * len(X_raw))
    train_size = len(X_raw) - val_size

    perm = torch.randperm(len(X_raw))
    X_train_t, X_val_t = X_raw[perm[:train_size]], X_raw[perm[train_size:]]
    y_train_t, y_val_t = y_raw[perm[:train_size]], y_raw[perm[train_size:]]

    model = MLP().to(device)
    model.load_state_dict(torch.load("mlp_model.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        orig_test_logits = model(X_test_t)
        orig_test_preds = orig_test_logits.argmax(dim=1)
        train_teacher_logits = model(X_train_t)
        val_teacher_logits = model(X_val_t)
        teacher_acc = (orig_test_preds == y_test_t).float().mean().item() * 100.0

    # Profiling on Validation set ONLY
    with torch.no_grad():
        h1_act = model.relu1(model.fc1(X_val_t)).cpu().numpy()
        h2_act = (
            model.relu2(model.fc2(torch.tensor(h1_act, device=device))).cpu().numpy()
        )

    fp_l1, fp_l2 = h1_act.T, h2_act.T
    mean_act_l1, mean_act_l2 = fp_l1.mean(axis=1), fp_l2.mean(axis=1)

    fp_l1_safe, fp_l2_safe = make_safe_fp(fp_l1), make_safe_fp(fp_l2)

    scores_l1, scores_l2 = np.zeros(256), np.zeros(128)
    with torch.no_grad():
        for i in range(256):
            h1 = model.relu1(model.fc1(X_val_t))
            h1[:, i] = 0.0
            scores_l1[i] = (
                (
                    1.0
                    - F.cosine_similarity(
                        val_teacher_logits, model.fc3(model.relu2(model.fc2(h1))), dim=1
                    )
                )
                .mean()
                .item()
            )

        for j in range(128):
            h1 = model.relu1(model.fc1(X_val_t))
            h2 = model.relu2(model.fc2(h1))
            h2[:, j] = 0.0
            scores_l2[j] = (
                (1.0 - F.cosine_similarity(val_teacher_logits, model.fc3(h2), dim=1))
                .mean()
                .item()
            )

    def fine_tune(sub_model):
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
            k_c1, k_c2 = max(1, target_budget_neurons[0]), max(
                1, target_budget_neurons[1]
            )
            top_l1, top_l2 = set(), set()

        rem_l1 = np.array([i for i in range(256) if i not in top_l1])
        rem_l2 = np.array([j for j in range(128) if j not in top_l2])

        W1, b1 = model.fc1.weight.detach().clone(), model.fc1.bias.detach().clone()
        W2, b2 = model.fc2.weight.detach().clone(), model.fc2.bias.detach().clone()
        W3, b3 = model.fc3.weight.detach().clone(), model.fc3.bias.detach().clone()

        kept_l1, kept_l2 = set(top_l1), set(top_l2)

        if len(rem_l1) > 0 and k_c1 > 0:
            c1 = AgglomerativeClustering(
                n_clusters=min(k_c1, len(rem_l1)), metric="cosine", linkage="average"
            )
            labels_l1 = c1.fit_predict(fp_l1_safe[rem_l1])
            for cid in range(k_c1):
                m = rem_l1[labels_l1 == cid]
                if len(m) == 0:
                    continue
                rep = find_active_cluster_medoid(fp_l1_safe, mean_act_l1, m)
                kept_l1.add(rep)
                W1[rep], b1[rep] = W1[m].mean(dim=0), b1[m].mean(dim=0)
                norm = np.sqrt(len(m))
                for x in m:
                    if x != rep:
                        W2[:, rep] += W2[:, x] / norm

        if len(rem_l2) > 0 and k_c2 > 0:
            c2 = AgglomerativeClustering(
                n_clusters=min(k_c2, len(rem_l2)), metric="cosine", linkage="average"
            )
            labels_l2 = c2.fit_predict(fp_l2_safe[rem_l2])
            for cid in range(k_c2):
                m = rem_l2[labels_l2 == cid]
                if len(m) == 0:
                    continue
                rep = find_active_cluster_medoid(fp_l2_safe, mean_act_l2, m)
                kept_l2.add(rep)
                W2[rep], b2[rep] = W2[m].mean(dim=0), b2[m].mean(dim=0)
                norm = np.sqrt(len(m))
                for x in m:
                    if x != rep:
                        W3[:, rep] += W3[:, x] / norm

        m_l1 = np.array([i in kept_l1 for i in range(256)])
        m_l2 = np.array([j in kept_l2 for j in range(128)])
        sub_w1, sub_b1 = W1[m_l1, :], b1[m_l1]
        sub_w2, sub_b2 = W2[m_l2, :][:, m_l1], b2[m_l2]
        sub_w3, sub_b3 = W3[:, m_l2], b3

        sub_model = AbstractSubMLP(sub_w1, sub_b1, sub_w2, sub_b2, sub_w3, sub_b3).to(
            device
        )

        with torch.no_grad():
            preds = sub_model(X_test_t).argmax(dim=1)
            zs_acc = (preds == y_test_t).float().mean().item() * 100.0
            zs_agr = (preds == orig_test_preds).float().mean().item() * 100.0

        t_zs = (time.time() - t0_zs) * 1000.0
        t0_ft = time.time()
        no_ema_acc, no_ema_agr, ema_acc, ema_agr = fine_tune(sub_model)
        t_ft = (time.time() - t0_ft) * 1000.0

        return (
            zs_acc,
            zs_agr,
            no_ema_acc,
            no_ema_agr,
            ema_acc,
            ema_agr,
            len(kept_l1),
            len(kept_l2),
            t_zs,
            t_ft,
        )

    seed_res = {
        "our_zs_acc": [],
        "our_zs_agr": [],
        "our_noema_acc": [],
        "our_noema_agr": [],
        "our_ema_acc": [],
        "our_ema_agr": [],
        "da_zs_acc": [],
        "da_zs_agr": [],
        "da_ft_acc": [],
        "da_ft_agr": [],
        "wf_zs_acc": [],
        "wf_zs_agr": [],
        "wf_ft_acc": [],
        "wf_ft_agr": [],
        "pct_kept": [],
        "our_zs_time": [],
        "our_ft_time": [],
        "da_zs_time": [],
        "da_ft_time": [],
        "wf_zs_time": [],
        "wf_ft_time": [],
        "teacher_acc": teacher_acc,
    }

    for b in target_budgets:
        (
            o_zs_acc,
            o_zs_agr,
            o_ne_acc,
            o_ne_agr,
            o_e_acc,
            o_e_agr,
            n1,
            n2,
            o_tzs,
            o_tft,
        ) = build_core(b, use_causal=True)
        d_zs_acc, d_zs_agr, _, _, d_e_acc, d_e_agr, _, _, d_tzs, d_tft = build_core(
            (n1, n2), use_causal=False
        )

        # Execute 6-step Global Causual Restructuring for current budget
        prune_pct = max(0.0, min(99.0, (1.0 - b / TOTAL_HIDDEN_NEURONS) * 100.0))
        wf_out = run_ablation_clustering_restructure_workflow(
            model,
            X_val_t,
            val_teacher_logits,
            top_n_pct=15.0,
            n_clusters=4,
            prune_pct_per_layer=prune_pct,
            device=device,
        )
        wf_net = wf_out["restructured_model"]

        # Zero-Shot Eval for Global Causual Restructuring
        wf_net.eval()
        t0_wf_zs = time.time()
        with torch.no_grad():
            wf_zs_preds = wf_net(X_test_t).argmax(dim=1)
            wf_zs_acc = (wf_zs_preds == y_test_t).float().mean().item() * 100.0
            wf_zs_agr = (wf_zs_preds == orig_test_preds).float().mean().item() * 100.0
        wf_tz = (time.time() - t0_wf_zs) * 1000.0

        # Fine-Tuning via KD + EMA for Global Causual Restructuring
        t0_wf_ft = time.time()
        _, _, wf_ft_acc, wf_ft_agr = fine_tune(wf_net)
        wf_tf = (time.time() - t0_wf_ft) * 1000.0

        pct = ((n1 + n2) / TOTAL_HIDDEN_NEURONS) * 100.0
        seed_res["pct_kept"].append(pct)
        seed_res["our_zs_acc"].append(o_zs_acc)
        seed_res["our_zs_agr"].append(o_zs_agr)
        seed_res["our_noema_acc"].append(o_ne_acc)
        seed_res["our_noema_agr"].append(o_ne_agr)
        seed_res["our_ema_acc"].append(o_e_acc)
        seed_res["our_ema_agr"].append(o_e_agr)
        seed_res["da_zs_acc"].append(d_zs_acc)
        seed_res["da_zs_agr"].append(d_zs_agr)
        seed_res["da_ft_acc"].append(d_e_acc)
        seed_res["da_ft_agr"].append(d_e_agr)
        seed_res["wf_zs_acc"].append(wf_zs_acc)
        seed_res["wf_zs_agr"].append(wf_zs_agr)
        seed_res["wf_ft_acc"].append(wf_ft_acc)
        seed_res["wf_ft_agr"].append(wf_ft_agr)
        seed_res["our_zs_time"].append(o_tzs)
        seed_res["our_ft_time"].append(o_tft)
        seed_res["da_zs_time"].append(d_tzs)
        seed_res["da_ft_time"].append(d_tft)
        seed_res["wf_zs_time"].append(wf_tz)
        seed_res["wf_ft_time"].append(wf_tf)

    return seed_res


# =====================================================================
# 4. Main Benchmark Orchestrator & Aggregation
# =====================================================================
print(f"=== RUNNING MULTI-SEED BENCHMARK (N={len(SEEDS)}) ===")
all_runs = [run_single_seed_benchmark(s) for s in SEEDS]

pct_axis = np.mean([r["pct_kept"] for r in all_runs], axis=0)


def aggregate_metric(key):
    data = np.array([r[key] for r in all_runs])
    return np.mean(data, axis=0), np.std(data, axis=0)


our_ema_acc_m, our_ema_acc_s = aggregate_metric("our_ema_acc")
our_ema_agr_m, our_ema_agr_s = aggregate_metric("our_ema_agr")
our_noema_acc_m, our_noema_acc_s = aggregate_metric("our_noema_acc")
our_noema_agr_m, our_noema_agr_s = aggregate_metric("our_noema_agr")
our_zs_acc_m, our_zs_acc_s = aggregate_metric("our_zs_acc")
our_zs_agr_m, our_zs_agr_s = aggregate_metric("our_zs_agr")

da_ft_acc_m, da_ft_acc_s = aggregate_metric("da_ft_acc")
da_ft_agr_m, da_ft_agr_s = aggregate_metric("da_ft_agr")
da_zs_acc_m, da_zs_acc_s = aggregate_metric("da_zs_acc")
da_zs_agr_m, da_zs_agr_s = aggregate_metric("da_zs_agr")

wf_ft_acc_m, wf_ft_acc_s = aggregate_metric("wf_ft_acc")
wf_ft_agr_m, wf_ft_agr_s = aggregate_metric("wf_ft_agr")
wf_zs_acc_m, wf_zs_acc_s = aggregate_metric("wf_zs_acc")
wf_zs_agr_m, wf_zs_agr_s = aggregate_metric("wf_zs_agr")

teacher_acc_m = np.mean([r["teacher_acc"] for r in all_runs])
teacher_acc_s = np.std([r["teacher_acc"] for r in all_runs])

# Print Summary Table
print("\n" + "=" * 95)
print(
    f"{'Budget (%)':<10} | {'Our FT Acc (%)':<20} | {'WF FT Acc (%)':<20} | {'DA FT Acc (%)':<20} | {'Delta (Our vs DA)':<18}"
)
print("=" * 95)
for i in range(len(pct_axis)):
    print(
        f"{pct_axis[i]:<10.1f} | {our_ema_acc_m[i]:.2f} ± {our_ema_acc_s[i]:.2f}{'':<8} | {wf_ft_acc_m[i]:.2f} ± {wf_ft_acc_s[i]:.2f}{'':<8} | {da_ft_acc_m[i]:.2f} ± {da_ft_acc_s[i]:.2f}{'':<8} | {our_ema_acc_m[i] - da_ft_acc_m[i]:+.2f}"
    )
print("=" * 95)

# =====================================================================
# 5. Plotting Figure with Multi-Seed Variance Bands
# =====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))


def plot_with_band(ax, x, mean, std, fmt, color, label):
    ax.plot(x, mean, fmt, color=color, linewidth=2.0, label=label)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)


# Panel 1: Agreement
plot_with_band(
    ax1,
    pct_axis,
    our_ema_agr_m,
    our_ema_agr_s,
    "D-",
    "#008080",
    "Layer-Wise Pruning (KD FT + EMA)",
)
plot_with_band(
    ax1,
    pct_axis,
    our_zs_agr_m,
    our_zs_agr_s,
    "s-.",
    "#1f77b4",
    "Layer-Wise Pruning (Zero-Shot)",
)
plot_with_band(
    ax1,
    pct_axis,
    wf_ft_agr_m,
    wf_ft_agr_s,
    "o-",
    "#d95f02",
    "Global Causual Restructuring (KD FT + EMA)",
)
plot_with_band(
    ax1,
    pct_axis,
    wf_zs_agr_m,
    wf_zs_agr_s,
    "x-.",
    "#e7298a",
    "Global Causual Restructuring (Zero-Shot)",
)
plot_with_band(
    ax1, pct_axis, da_ft_agr_m, da_ft_agr_s, "^-", "#6a3d9a", "DeepAbstract (KD FT)"
)
plot_with_band(
    ax1,
    pct_axis,
    da_zs_agr_m,
    da_zs_agr_s,
    "v-.",
    "#9467bd",
    "DeepAbstract (Zero-Shot)",
)

ax1.set_xlabel("Percentage of Neurons Kept (%)", fontsize=11)
ax1.set_ylabel("Agreement with Original Model (%)", fontsize=11)
ax1.set_title("Agreement vs. % Neurons Kept (Mean ± Std)", fontsize=13)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(fontsize=8, loc="lower right")

# Panel 2: Test Accuracy
plot_with_band(
    ax2,
    pct_axis,
    our_ema_acc_m,
    our_ema_acc_s,
    "D-",
    "#008080",
    "Layer-Wise Pruning (KD FT + EMA)",
)
plot_with_band(
    ax2,
    pct_axis,
    our_zs_acc_m,
    our_zs_acc_s,
    "s-.",
    "#2ca02c",
    "Layer-Wise Pruning (Zero-Shot)",
)
plot_with_band(
    ax2,
    pct_axis,
    wf_ft_acc_m,
    wf_ft_acc_s,
    "o-",
    "#d95f02",
    "Global Causual Restructuring (KD FT + EMA)",
)
plot_with_band(
    ax2,
    pct_axis,
    wf_zs_acc_m,
    wf_zs_acc_s,
    "x-.",
    "#e7298a",
    "Global Causual Restructuring (Zero-Shot)",
)
plot_with_band(
    ax2, pct_axis, da_ft_acc_m, da_ft_acc_s, "^-", "#6a3d9a", "DeepAbstract (KD FT)"
)
plot_with_band(
    ax2,
    pct_axis,
    da_zs_acc_m,
    da_zs_acc_s,
    "v-.",
    "#9467bd",
    "DeepAbstract (Zero-Shot)",
)

# Original model baseline
ax2.axhline(
    y=teacher_acc_m,
    color="#d62728",
    linestyle="--",
    linewidth=2.0,
    label=f"Original Model ({teacher_acc_m:.2f}%)",
)
ax2.fill_between(
    [pct_axis[0], pct_axis[-1]],
    teacher_acc_m - teacher_acc_s,
    teacher_acc_m + teacher_acc_s,
    color="#d62728",
    alpha=0.15,
)

ax2.set_xlabel("Percentage of Neurons Kept (%)", fontsize=11)
ax2.set_ylabel("Test Accuracy (%)", fontsize=11)
ax2.set_title("Test Accuracy vs. % Neurons Kept (Mean ± Std)", fontsize=13)
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(fontsize=8, loc="lower right")

plt.tight_layout()
plt.savefig("clean_benchmark_multiseed.png", dpi=300)
plt.close()

print("\nBenchmark complete. Output plot saved to 'clean_benchmark_multiseed.png'")
