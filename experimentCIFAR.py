import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import StratifiedShuffleSplit

# =====================================================================
# 0. Global Setup & Device Routing
# =====================================================================
# SEEDS = [42, 101, 2024, 7, 99]
SEEDS = [42]
TOTAL_CHANNELS = 64 + 128 + 256  # 448 Total Conv Channels
target_budgets = [45, 90, 135, 180, 225, 270, 315, 360]  # Kept channel budgets
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GLOBAL_START_TIME = time.time()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_elapsed_str():
    elapsed = time.time() - GLOBAL_START_TIME
    mins, secs = divmod(int(elapsed), 60)
    return f"[{mins:02d}:{secs:02d}]"


def make_safe_fp(fp):
    """Prevents ValueError in cosine clustering by handling all-zero feature maps."""
    fp_safe = fp.copy()
    zero_mask = np.all(fp_safe == 0, axis=1)
    if np.any(zero_mask):
        fp_safe[zero_mask] = 1e-8
    return fp_safe


# =====================================================================
# 1. Vision Architecture & Sub-Network Module
# =====================================================================
class CIFARConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        h1 = self.pool1(self.relu1(self.conv1(x)))
        h2 = self.pool2(self.relu2(self.conv2(h1)))
        h3 = self.pool3(self.relu3(self.conv3(h2)))
        return self.fc(torch.flatten(h3, 1))


class AbstractSubConvNet(nn.Module):
    def __init__(self, w1, b1, w2, b2, w3, b3, w_fc, b_fc):
        super().__init__()
        self.conv1 = nn.Conv2d(3, w1.shape[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(w2.shape[1], w2.shape[0], kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(w3.shape[1], w3.shape[0], kernel_size=3, padding=1)
        self.fc = nn.Linear(w_fc.shape[1], w_fc.shape[0])

        self.conv1.weight, self.conv1.bias = nn.Parameter(w1.clone()), nn.Parameter(
            b1.clone()
        )
        self.conv2.weight, self.conv2.bias = nn.Parameter(w2.clone()), nn.Parameter(
            b2.clone()
        )
        self.conv3.weight, self.conv3.bias = nn.Parameter(w3.clone()), nn.Parameter(
            b3.clone()
        )
        self.fc.weight, self.fc.bias = nn.Parameter(w_fc.clone()), nn.Parameter(
            b_fc.clone()
        )

    def forward(self, x):
        h1 = F.max_pool2d(F.relu(self.conv1(x)), 2, 2)
        h2 = F.max_pool2d(F.relu(self.conv2(h1)), 2, 2)
        h3 = F.adaptive_avg_pool2d(F.relu(self.conv3(h2)), (1, 1))
        return self.fc(torch.flatten(h3, 1))


class EMAModel:
    def __init__(self, model, decay=0.98):
        self.model = model
        self.decay = decay
        self.shadow = {
            n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad
        }
        self.backup = {}

    def update(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = (1.0 - self.decay) * p.data + self.decay * self.shadow[
                    n
                ]

    def apply_shadow(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    def restore(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.backup[n])


# =====================================================================
# 2. Helpers: Channel Medoid & Distillation Loss
# =====================================================================
def find_channel_medoid(fps, mean_acts, members):
    if len(members) == 1:
        return members[0]
    sub_fps = fps[members]
    centroid = sub_fps.mean(axis=0)
    dists = np.array(
        [
            1.0
            - np.dot(fp, centroid)
            / (np.linalg.norm(fp) * np.linalg.norm(centroid) + 1e-8)
            for fp in sub_fps
        ]
    )
    penalty = np.where(mean_acts[members] < 1e-5, 10.0, 0.0)
    return members[np.argmin(dists + penalty)]


def kd_loss_fn(s_logits, t_logits, labels, T=2.0, alpha=0.85):
    soft_t = F.softmax(t_logits / T, dim=1)
    soft_s = F.log_softmax(s_logits / T, dim=1)
    kl = F.kl_div(soft_s, soft_t, reduction="batchmean") * (T**2)
    ce = F.cross_entropy(s_logits, labels)
    return alpha * kl + (1.0 - alpha) * ce


# =====================================================================
# 2b. Workflow Method: Ablation, Fingerprinting, Clustering & Restructuring
# =====================================================================
def run_ablation_clustering_restructure_workflow(
    teacher_model,
    val_loader,
    top_n_pct=15.0,
    n_clusters=4,
    prune_pct_per_layer=20.0,
    device=device,
):
    """
    Executes the 6-step workflow:
    1. Single Neuron Ablation
    2. Behavioral Fingerprinting
    3. Keep top n% of neurons
    4. Agglomerative Clustering of (100-n)% of neurons
    5. Restructure network such that each cluster is a layer and layer 1 is the top n% of the neurons.
    6. Remove the least important (by ablation) neurons from each layer
    """
    teacher_model.eval()
    val_imgs = torch.cat([x.to(device) for x, _ in val_loader])

    # -----------------------------------------------------------------
    # Step 1 & Step 2: Behavioral Fingerprinting & Causal Scoring
    # -----------------------------------------------------------------
    with torch.no_grad():
        val_teacher_logits = teacher_model(val_imgs)
        c1_act = teacher_model.pool1(teacher_model.relu1(teacher_model.conv1(val_imgs)))
        c2_act = teacher_model.pool2(teacher_model.relu2(teacher_model.conv2(c1_act)))
        c3_act = teacher_model.pool3(teacher_model.relu3(teacher_model.conv3(c2_act)))

    fp_c1 = c1_act.mean(dim=(2, 3)).T.cpu().numpy()
    fp_c2 = c2_act.mean(dim=(2, 3)).T.cpu().numpy()
    fp_c3 = c3_act.mean(dim=(2, 3)).T.cpu().numpy()

    all_fps = np.vstack([fp_c1, fp_c2, fp_c3])
    all_fps_safe = make_safe_fp(all_fps)

    neuron_map = []
    for i in range(64):
        neuron_map.append(("conv1", i))
    for i in range(128):
        neuron_map.append(("conv2", i))
    for i in range(256):
        neuron_map.append(("conv3", i))

    ablation_scores = np.zeros(len(neuron_map))

    with torch.no_grad():
        for i in range(64):
            temp_c1 = c1_act.clone()
            temp_c1[:, i, :, :] = 0.0
            out = teacher_model.fc(
                torch.flatten(
                    teacher_model.pool3(
                        teacher_model.relu3(
                            teacher_model.conv3(
                                teacher_model.pool2(
                                    teacher_model.relu2(teacher_model.conv2(temp_c1))
                                )
                            )
                        )
                    ),
                    1,
                )
            )
            ablation_scores[i] = (
                (1.0 - F.cosine_similarity(val_teacher_logits, out, dim=1))
                .mean()
                .item()
            )

        for j in range(128):
            temp_c2 = c2_act.clone()
            temp_c2[:, j, :, :] = 0.0
            out = teacher_model.fc(
                torch.flatten(
                    teacher_model.pool3(
                        teacher_model.relu3(teacher_model.conv3(temp_c2))
                    ),
                    1,
                )
            )
            ablation_scores[64 + j] = (
                (1.0 - F.cosine_similarity(val_teacher_logits, out, dim=1))
                .mean()
                .item()
            )

        for k in range(256):
            temp_c3 = c3_act.clone()
            temp_c3[:, k, :, :] = 0.0
            out = teacher_model.fc(torch.flatten(temp_c3, 1))
            ablation_scores[64 + 128 + k] = (
                (1.0 - F.cosine_similarity(val_teacher_logits, out, dim=1))
                .mean()
                .item()
            )

    # -----------------------------------------------------------------
    # Step 3: Keep top n% of neurons
    # -----------------------------------------------------------------
    total_neurons = len(neuron_map)
    n_top = int(np.ceil((top_n_pct / 100.0) * total_neurons))
    sorted_indices = np.argsort(ablation_scores)[::-1]  # Descending order by score
    top_n_indices = sorted_indices[:n_top]
    rem_indices = sorted_indices[n_top:]

    # -----------------------------------------------------------------
    # Step 4: Agglomerative Clustering of (100-n)% of neurons
    # -----------------------------------------------------------------
    rem_fps = all_fps_safe[rem_indices]
    actual_clusters = min(n_clusters, len(rem_indices))
    clustering = AgglomerativeClustering(
        n_clusters=actual_clusters, metric="cosine", linkage="average"
    )
    cluster_labels = clustering.fit_predict(rem_fps)

    # -----------------------------------------------------------------
    # Step 5: Restructure network such that each cluster is a layer and layer 1 is top n%
    # -----------------------------------------------------------------
    restructured_layers = {}
    restructured_layers[1] = top_n_indices.tolist()

    for c_id in range(actual_clusters):
        cluster_members = rem_indices[cluster_labels == c_id].tolist()
        restructured_layers[c_id + 2] = cluster_members

    # -----------------------------------------------------------------
    # Step 6: Remove least important (by ablation) neurons from each layer
    # -----------------------------------------------------------------
    pruned_layers = {}
    for layer_idx, neurons in restructured_layers.items():
        if len(neurons) == 0:
            pruned_layers[layer_idx] = []
            continue
        layer_scores = ablation_scores[neurons]
        sorted_in_layer = np.argsort(
            layer_scores
        )  # Ascending order (least important first)
        n_remove = int(np.floor(len(neurons) * (prune_pct_per_layer / 100.0)))
        kept_in_layer = [neurons[idx] for idx in sorted_in_layer[n_remove:]]
        pruned_layers[layer_idx] = kept_in_layer

    class RestructuredClusterNet(nn.Module):
        def __init__(self, teacher_model, pruned_layers, neuron_map):
            super().__init__()
            self.pruned_layers = pruned_layers
            self.neuron_map = neuron_map

            all_kept = []
            for layer_idx in sorted(pruned_layers.keys()):
                all_kept.extend(pruned_layers[layer_idx])

            kept1 = [i for i in all_kept if i < 64]
            kept2 = [i - 64 for i in all_kept if 64 <= i < 192]
            kept3 = [i - 192 for i in all_kept if i >= 192]

            if len(kept1) == 0:
                kept1 = [0]
            if len(kept2) == 0:
                kept2 = [0]
            if len(kept3) == 0:
                kept3 = [0]

            w1 = teacher_model.conv1.weight.detach()[kept1, :, :, :]
            b1 = teacher_model.conv1.bias.detach()[kept1]

            w2 = teacher_model.conv2.weight.detach()[kept2, :, :, :][:, kept1, :, :]
            b2 = teacher_model.conv2.bias.detach()[kept2]

            w3 = teacher_model.conv3.weight.detach()[kept3, :, :, :][:, kept2, :, :]
            b3 = teacher_model.conv3.bias.detach()[kept3]

            w_fc = teacher_model.fc.weight.detach()[:, kept3]
            b_fc = teacher_model.fc.bias.detach()

            self.conv1 = nn.Conv2d(3, len(kept1), kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(len(kept1), len(kept2), kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(len(kept2), len(kept3), kernel_size=3, padding=1)
            self.fc = nn.Linear(len(kept3), 10)

            self.conv1.weight, self.conv1.bias = nn.Parameter(w1.clone()), nn.Parameter(
                b1.clone()
            )
            self.conv2.weight, self.conv2.bias = nn.Parameter(w2.clone()), nn.Parameter(
                b2.clone()
            )
            self.conv3.weight, self.conv3.bias = nn.Parameter(w3.clone()), nn.Parameter(
                b3.clone()
            )
            self.fc.weight, self.fc.bias = nn.Parameter(w_fc.clone()), nn.Parameter(
                b_fc.clone()
            )

        def forward(self, x):
            h1 = F.max_pool2d(F.relu(self.conv1(x)), 2, 2)
            h2 = F.max_pool2d(F.relu(self.conv2(h1)), 2, 2)
            h3 = F.adaptive_avg_pool2d(F.relu(self.conv3(h2)), (1, 1))
            return self.fc(torch.flatten(h3, 1))

    restructured_net = RestructuredClusterNet(
        teacher_model, pruned_layers, neuron_map
    ).to(device)

    return {
        "ablation_scores": ablation_scores,
        "top_n_indices": top_n_indices,
        "restructured_layers": restructured_layers,
        "pruned_layers": pruned_layers,
        "restructured_model": restructured_net,
    }


# =====================================================================
# 3. Single-Seed Benchmark Execution
# =====================================================================
def run_cifar_seed_benchmark(seed_idx, seed):
    set_seed(seed)
    print(
        f"\n{get_elapsed_str()} === Starting Seed [{seed_idx + 1}/{len(SEEDS)}] (Seed: {seed}) ==="
    )

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    full_train = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    # Balanced (Stratified) 80/20 Train/Val Split (40k Train / 10k Val)
    targets = np.array(full_train.targets)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))

    train_loader = DataLoader(
        Subset(full_train, train_idx), batch_size=128, shuffle=True
    )
    val_loader = DataLoader(Subset(full_train, val_idx), batch_size=128, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    teacher_model = CIFARConvNet().to(device)

    # -----------------------------------------------------------------
    # Step 1/3: Load Latest Checkpoint
    # -----------------------------------------------------------------
    ckpt_dir = "./checkpoints"
    latest_path = os.path.join(ckpt_dir, f"teacher_seed_{seed}_latest.pth")
    ckpt_path = None

    if os.path.exists(latest_path):
        ckpt_path = latest_path
    elif os.path.exists(os.path.join(ckpt_dir, f"teacher_seed_{seed}.pth")):
        ckpt_path = os.path.join(ckpt_dir, f"teacher_seed_{seed}.pth")
    elif os.path.exists(ckpt_dir):
        epoch_files = [
            f
            for f in os.listdir(ckpt_dir)
            if f.startswith(f"teacher_seed_{seed}_epoch_") and f.endswith(".pth")
        ]
        if epoch_files:
            epoch_files.sort(key=lambda x: int(x.split("_epoch_")[1].split(".pth")[0]))
            ckpt_path = os.path.join(ckpt_dir, epoch_files[-1])

    if ckpt_path and os.path.exists(ckpt_path):
        print(
            f"{get_elapsed_str()}   [Step 1/3] Loading Latest Teacher Checkpoint ({ckpt_path})..."
        )
        ckpt = torch.load(ckpt_path, map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            teacher_model.load_state_dict(ckpt["state_dict"])
            print(
                f"{get_elapsed_str()}     -> Successfully loaded state dict from Epoch {ckpt.get('epoch', 'N/A')}"
            )
        else:
            teacher_model.load_state_dict(ckpt)
            print(f"{get_elapsed_str()}     -> Successfully loaded state dict.")
    else:
        print(
            f"{get_elapsed_str()}   [Step 1/3] No checkpoint found. Training Teacher Baseline (15 Epochs)..."
        )
        optimizer = optim.Adam(teacher_model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        teacher_model.train()
        for epoch in range(15):
            running_loss = 0.0
            for x_b, y_b in train_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                out = teacher_model(x_b)
                loss = criterion(out, y_b)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            if (epoch + 1) % 5 == 0 or epoch == 14:
                print(
                    f"{get_elapsed_str()}     -> Teacher Epoch {epoch+1:02d}/15 | Loss: {running_loss/len(train_loader):.4f}"
                )

    teacher_model.eval()

    test_imgs, test_targets = [], []
    for x_b, y_b in test_loader:
        test_imgs.append(x_b)
        test_targets.append(y_b)
    X_test = torch.cat(test_imgs).to(device)
    y_test = torch.cat(test_targets).to(device)

    with torch.no_grad():
        teacher_acc = (
            teacher_model(X_test).argmax(dim=1) == y_test
        ).float().mean().item() * 100.0
        orig_test_preds = teacher_model(X_test).argmax(dim=1)

    print(f"{get_elapsed_str()}   [Teacher Accuracy]: {teacher_acc:.2f}%")

    # -----------------------------------------------------------------
    # Causal Channel Profiling on VALIDATION SET ONLY
    # -----------------------------------------------------------------
    print(
        f"{get_elapsed_str()}   [Step 2/3] Profiling Causal Channels & Computing Fingerprints..."
    )
    val_imgs = torch.cat([x.to(device) for x, _ in val_loader])
    with torch.no_grad():
        val_teacher_logits = teacher_model(val_imgs)
        c1_act = teacher_model.pool1(teacher_model.relu1(teacher_model.conv1(val_imgs)))
        c2_act = teacher_model.pool2(teacher_model.relu2(teacher_model.conv2(c1_act)))
        c3_act = teacher_model.pool3(teacher_model.relu3(teacher_model.conv3(c2_act)))

    fp_c1 = c1_act.mean(dim=(2, 3)).T.cpu().numpy()
    fp_c2 = c2_act.mean(dim=(2, 3)).T.cpu().numpy()
    fp_c3 = c3_act.mean(dim=(2, 3)).T.cpu().numpy()

    # Apply fix for zero-vector cosine distance issue
    fp_c1_safe = make_safe_fp(fp_c1)
    fp_c2_safe = make_safe_fp(fp_c2)
    fp_c3_safe = make_safe_fp(fp_c3)

    mean_act_c1, mean_act_c2, mean_act_c3 = (
        fp_c1.mean(axis=1),
        fp_c2.mean(axis=1),
        fp_c3.mean(axis=1),
    )

    scores_c1 = np.zeros(64)
    scores_c2 = np.zeros(128)
    scores_c3 = np.zeros(256)

    # Channel Ablation Causal Scoring
    with torch.no_grad():
        for i in range(64):
            temp_c1 = c1_act.clone()
            temp_c1[:, i, :, :] = 0.0
            out = teacher_model.fc(
                torch.flatten(
                    teacher_model.pool3(
                        teacher_model.relu3(
                            teacher_model.conv3(
                                teacher_model.pool2(
                                    teacher_model.relu2(teacher_model.conv2(temp_c1))
                                )
                            )
                        )
                    ),
                    1,
                )
            )
            scores_c1[i] = (
                (1.0 - F.cosine_similarity(val_teacher_logits, out, dim=1))
                .mean()
                .item()
            )

        for j in range(128):
            temp_c2 = c2_act.clone()
            temp_c2[:, j, :, :] = 0.0
            out = teacher_model.fc(
                torch.flatten(
                    teacher_model.pool3(
                        teacher_model.relu3(teacher_model.conv3(temp_c2))
                    ),
                    1,
                )
            )
            scores_c2[j] = (
                (1.0 - F.cosine_similarity(val_teacher_logits, out, dim=1))
                .mean()
                .item()
            )

        for k in range(256):
            temp_c3 = c3_act.clone()
            temp_c3[:, k, :, :] = 0.0
            out = teacher_model.fc(torch.flatten(temp_c3, 1))
            scores_c3[k] = (
                (1.0 - F.cosine_similarity(val_teacher_logits, out, dim=1))
                .mean()
                .item()
            )

    # -----------------------------------------------------------------
    # Benchmark Target Budgets
    # -----------------------------------------------------------------
    print(f"{get_elapsed_str()}   [Step 3/3] Running Abstraction Budgets...")

    def build_conv_core(target_channels, use_causal=True):
        t0_zs = time.time()
        b1 = max(1, int(round(target_channels * (64 / 448))))
        b2 = max(1, int(round(target_channels * (128 / 448))))
        b3 = max(1, target_channels - b1 - b2)

        if use_causal:
            res_ratio = 0.10 if target_channels <= 150 else 0.15
            n_top1, n_top2, n_top3 = (
                int(b1 * res_ratio),
                int(b2 * res_ratio),
                int(b3 * res_ratio),
            )
            k_c1, k_c2, k_c3 = (
                max(1, b1 - n_top1),
                max(1, b2 - n_top2),
                max(1, b3 - n_top3),
            )
            top1 = set(np.argsort(scores_c1)[-n_top1:]) if n_top1 > 0 else set()
            top2 = set(np.argsort(scores_c2)[-n_top2:]) if n_top2 > 0 else set()
            top3 = set(np.argsort(scores_c3)[-n_top3:]) if n_top3 > 0 else set()
        else:
            k_c1, k_c2, k_c3 = b1, b2, b3
            top1, top2, top3 = set(), set(), set()

        rem1 = np.array([i for i in range(64) if i not in top1])
        rem2 = np.array([j for j in range(128) if j not in top2])
        rem3 = np.array([k for k in range(256) if k not in top3])

        W1, b1_t = (
            teacher_model.conv1.weight.detach().clone(),
            teacher_model.conv1.bias.detach().clone(),
        )
        W2, b2_t = (
            teacher_model.conv2.weight.detach().clone(),
            teacher_model.conv2.bias.detach().clone(),
        )
        W3, b3_t = (
            teacher_model.conv3.weight.detach().clone(),
            teacher_model.conv3.bias.detach().clone(),
        )
        W_fc, b_fc = (
            teacher_model.fc.weight.detach().clone(),
            teacher_model.fc.bias.detach().clone(),
        )

        kept1, kept2, kept3 = set(top1), set(top2), set(top3)

        # Layer 1 Clustering
        if len(rem1) > 0 and k_c1 > 0:
            c1 = AgglomerativeClustering(
                n_clusters=min(k_c1, len(rem1)), metric="cosine", linkage="average"
            )
            labels1 = c1.fit_predict(fp_c1_safe[rem1])
            for cid in range(k_c1):
                m = rem1[labels1 == cid]
                if len(m) == 0:
                    continue
                rep = find_channel_medoid(fp_c1_safe, mean_act_c1, m)
                kept1.add(rep)
                W1[rep] = W1[m].mean(dim=0)
                b1_t[rep] = b1_t[m].mean(dim=0)
                norm = np.sqrt(len(m))
                for x in m:
                    if x != rep:
                        W2[:, rep, :, :] += W2[:, x, :, :] / norm

        # Layer 2 Clustering
        if len(rem2) > 0 and k_c2 > 0:
            c2 = AgglomerativeClustering(
                n_clusters=min(k_c2, len(rem2)), metric="cosine", linkage="average"
            )
            labels2 = c2.fit_predict(fp_c2_safe[rem2])
            for cid in range(k_c2):
                m = rem2[labels2 == cid]
                if len(m) == 0:
                    continue
                rep = find_channel_medoid(fp_c2_safe, mean_act_c2, m)
                kept2.add(rep)
                W2[rep] = W2[m].mean(dim=0)
                b2_t[rep] = b2_t[m].mean(dim=0)
                norm = np.sqrt(len(m))
                for x in m:
                    if x != rep:
                        W3[:, rep, :, :] += W3[:, x, :, :] / norm

        # Layer 3 Clustering
        if len(rem3) > 0 and k_c3 > 0:
            c3 = AgglomerativeClustering(
                n_clusters=min(k_c3, len(rem3)), metric="cosine", linkage="average"
            )
            labels3 = c3.fit_predict(fp_c3_safe[rem3])
            for cid in range(k_c3):
                m = rem3[labels3 == cid]
                if len(m) == 0:
                    continue
                rep = find_channel_medoid(fp_c3_safe, mean_act_c3, m)
                kept3.add(rep)
                W3[rep] = W3[m].mean(dim=0)
                b3_t[rep] = b3_t[m].mean(dim=0)
                norm = np.sqrt(len(m))
                for x in m:
                    if x != rep:
                        W_fc[:, rep] += W_fc[:, x] / norm

        m1, m2, m3 = (
            np.array([i in kept1 for i in range(64)]),
            np.array([j in kept2 for j in range(128)]),
            np.array([k in kept3 for k in range(256)]),
        )

        sub_w1, sub_b1 = W1[m1, :, :, :], b1_t[m1]
        sub_w2, sub_b2 = W2[m2, :, :, :][:, m1, :, :], b2_t[m2]
        sub_w3, sub_b3 = W3[m3, :, :, :][:, m2, :, :], b3_t[m3]
        sub_wfc, sub_bfc = W_fc[:, m3], b_fc

        sub_net = AbstractSubConvNet(
            sub_w1, sub_b1, sub_w2, sub_b2, sub_w3, sub_b3, sub_wfc, sub_bfc
        ).to(device)

        # Zero-Shot Eval
        sub_net.eval()
        with torch.no_grad():
            zs_logits = sub_net(X_test)
            zs_preds = zs_logits.argmax(dim=1)
            zs_acc = (zs_preds == y_test).float().mean().item() * 100.0
            zs_agr = (zs_preds == orig_test_preds).float().mean().item() * 100.0
        t_zs = (time.time() - t0_zs) * 1000.0

        # Fine-Tuning via KD + EMA
        t0_ft = time.time()
        opt = optim.Adam(sub_net.parameters(), lr=1e-3)
        ema = EMAModel(sub_net, decay=0.98)
        sub_net.train()

        for epoch in range(5):
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                with torch.no_grad():
                    t_logits = teacher_model(bx)
                opt.zero_grad()
                s_logits = sub_net(bx)
                loss = kd_loss_fn(s_logits, t_logits, by)
                loss.backward()
                opt.step()
                ema.update()

        ema.apply_shadow()
        sub_net.eval()
        with torch.no_grad():
            ft_preds = sub_net(X_test).argmax(dim=1)
            ft_acc = (ft_preds == y_test).float().mean().item() * 100.0
            ft_agr = (ft_preds == orig_test_preds).float().mean().item() * 100.0
        ema.restore()
        t_ft = (time.time() - t0_ft) * 1000.0

        return (
            zs_acc,
            zs_agr,
            ft_acc,
            ft_agr,
            len(kept1) + len(kept2) + len(kept3),
            t_zs,
            t_ft,
        )

    res = {
        "our_zs_acc": [],
        "our_zs_agr": [],
        "our_ft_acc": [],
        "our_ft_agr": [],
        "da_zs_acc": [],
        "da_zs_agr": [],
        "da_ft_acc": [],
        "da_ft_agr": [],
        "wf_zs_acc": [],
        "wf_zs_agr": [],
        "wf_ft_acc": [],
        "wf_ft_agr": [],
        "pct_kept": [],
        "our_zs_t": [],
        "our_ft_t": [],
        "da_zs_t": [],
        "da_ft_t": [],
        "wf_zs_t": [],
        "wf_ft_t": [],
        "teacher_acc": teacher_acc,
    }

    for b in target_budgets:
        t_b_start = time.time()
        o_za, o_zg, o_fa, o_fg, total_k, o_tz, o_tf = build_conv_core(
            b, use_causal=True
        )
        d_za, d_zg, d_fa, d_fg, _, d_tz, d_tf = build_conv_core(
            total_k, use_causal=False
        )

        # Execute 6-step workflow method for current budget
        prune_pct = max(0.0, min(99.0, (1.0 - b / TOTAL_CHANNELS) * 100.0))
        wf_out = run_ablation_clustering_restructure_workflow(
            teacher_model,
            val_loader,
            top_n_pct=15.0,
            n_clusters=4,
            prune_pct_per_layer=prune_pct,
            device=device,
        )
        wf_net = wf_out["restructured_model"]

        # Zero-Shot Eval for Workflow Method
        wf_net.eval()
        t0_wf_zs = time.time()
        with torch.no_grad():
            wf_zs_logits = wf_net(X_test)
            wf_zs_preds = wf_zs_logits.argmax(dim=1)
            wf_zs_acc = (wf_zs_preds == y_test).float().mean().item() * 100.0
            wf_zs_agr = (wf_zs_preds == orig_test_preds).float().mean().item() * 100.0
        wf_tz = (time.time() - t0_wf_zs) * 1000.0

        # Fine-Tuning via KD + EMA for Workflow Method
        t0_wf_ft = time.time()
        wf_opt = optim.Adam(wf_net.parameters(), lr=1e-3)
        wf_ema = EMAModel(wf_net, decay=0.98)
        wf_net.train()
        for epoch in range(5):
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                with torch.no_grad():
                    t_logits = teacher_model(bx)
                wf_opt.zero_grad()
                s_logits = wf_net(bx)
                loss = kd_loss_fn(s_logits, t_logits, by)
                loss.backward()
                wf_opt.step()
                wf_ema.update()

        wf_ema.apply_shadow()
        wf_net.eval()
        with torch.no_grad():
            wf_ft_preds = wf_net(X_test).argmax(dim=1)
            wf_ft_acc = (wf_ft_preds == y_test).float().mean().item() * 100.0
            wf_ft_agr = (wf_ft_preds == orig_test_preds).float().mean().item() * 100.0
        wf_ema.restore()
        wf_tf = (time.time() - t0_wf_ft) * 1000.0

        pct = (total_k / TOTAL_CHANNELS) * 100.0
        res["pct_kept"].append(pct)
        res["our_zs_acc"].append(o_za)
        res["our_zs_agr"].append(o_zg)
        res["our_ft_acc"].append(o_fa)
        res["our_ft_agr"].append(o_fg)
        res["da_zs_acc"].append(d_za)
        res["da_zs_agr"].append(d_zg)
        res["da_ft_acc"].append(d_fa)
        res["da_ft_agr"].append(d_fg)
        res["wf_zs_acc"].append(wf_zs_acc)
        res["wf_zs_agr"].append(wf_zs_agr)
        res["wf_ft_acc"].append(wf_ft_acc)
        res["wf_ft_agr"].append(wf_ft_agr)
        res["our_zs_t"].append(o_tz)
        res["our_ft_t"].append(o_tf)
        res["da_zs_t"].append(d_tz)
        res["da_ft_t"].append(d_tf)
        res["wf_zs_t"].append(wf_tz)
        res["wf_ft_t"].append(wf_tf)

        b_time = time.time() - t_b_start
        print(
            f"{get_elapsed_str()}     -> Budget {pct:4.1f}% ({total_k:3d} ch) | Our FT: {o_fa:.2f}% | WF FT: {wf_ft_acc:.2f}% | DA FT: {d_fa:.2f}% ({b_time:.1f}s)"
        )

    return res


# =====================================================================
# 4. Main Benchmark Orchestrator
# =====================================================================
print(f"=== STARTING CIFAR-10 BENCHMARK ({len(SEEDS)} SEEDS) ===")
all_cifar_runs = []
for idx, s in enumerate(SEEDS):
    all_cifar_runs.append(run_cifar_seed_benchmark(idx, s))

pct_axis = np.mean([r["pct_kept"] for r in all_cifar_runs], axis=0)


def aggregate_cifar(key):
    data = np.array([r[key] for r in all_cifar_runs])
    return np.mean(data, axis=0), np.std(data, axis=0)


our_ft_acc_m, our_ft_acc_s = aggregate_cifar("our_ft_acc")
our_ft_agr_m, our_ft_agr_s = aggregate_cifar("our_ft_agr")
our_zs_acc_m, our_zs_acc_s = aggregate_cifar("our_zs_acc")
our_zs_agr_m, our_zs_agr_s = aggregate_cifar("our_zs_agr")

da_ft_acc_m, da_ft_acc_s = aggregate_cifar("da_ft_acc")
da_ft_agr_m, da_ft_agr_s = aggregate_cifar("da_ft_agr")
da_zs_acc_m, da_zs_acc_s = aggregate_cifar("da_zs_acc")
da_zs_agr_m, da_zs_agr_s = aggregate_cifar("da_zs_agr")

wf_ft_acc_m, wf_ft_acc_s = aggregate_cifar("wf_ft_acc")
wf_ft_agr_m, wf_ft_agr_s = aggregate_cifar("wf_ft_agr")
wf_zs_acc_m, wf_zs_acc_s = aggregate_cifar("wf_zs_acc")
wf_zs_agr_m, wf_zs_agr_s = aggregate_cifar("wf_zs_agr")

teacher_acc_m = np.mean([r["teacher_acc"] for r in all_cifar_runs])
teacher_acc_s = np.std([r["teacher_acc"] for r in all_cifar_runs])

# =====================================================================
# 5. Plotting
# =====================================================================
print(f"\n{get_elapsed_str()} Generating Benchmark Figures...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))


def plot_cifar_band(ax, x, mean, std, fmt, color, label):
    ax.plot(x, mean, fmt, color=color, linewidth=2.0, label=label)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)


# Agreement Plot
plot_cifar_band(
    ax1,
    pct_axis,
    our_ft_agr_m,
    our_ft_agr_s,
    "D-",
    "#008080",
    "Our Method (KD FT + EMA)",
)
plot_cifar_band(
    ax1,
    pct_axis,
    our_zs_agr_m,
    our_zs_agr_s,
    "s-.",
    "#1f77b4",
    "Our Method (Zero-Shot)",
)
plot_cifar_band(
    ax1,
    pct_axis,
    wf_ft_agr_m,
    wf_ft_agr_s,
    "o-",
    "#d95f02",
    "Workflow Method (KD FT + EMA)",
)
plot_cifar_band(
    ax1,
    pct_axis,
    wf_zs_agr_m,
    wf_zs_agr_s,
    "x-.",
    "#e7298a",
    "Workflow Method (Zero-Shot)",
)
plot_cifar_band(
    ax1, pct_axis, da_ft_agr_m, da_ft_agr_s, "^-", "#6a3d9a", "DeepAbstract (KD FT)"
)
plot_cifar_band(
    ax1,
    pct_axis,
    da_zs_agr_m,
    da_zs_agr_s,
    "v-.",
    "#9467bd",
    "DeepAbstract (Zero-Shot)",
)
ax1.set_xlabel("Percentage of Conv Channels Kept (%)")
ax1.set_ylabel("Agreement with Teacher (%)")
ax1.set_title("CIFAR-10 Agreement vs. % Channels Kept")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="lower right", fontsize=8)

# Accuracy Plot with Unperturbed Baseline
plot_cifar_band(
    ax2,
    pct_axis,
    our_ft_acc_m,
    our_ft_acc_s,
    "D-",
    "#008080",
    "Our Method (KD FT + EMA)",
)
plot_cifar_band(
    ax2,
    pct_axis,
    our_zs_acc_m,
    our_zs_acc_s,
    "s-.",
    "#2ca02c",
    "Our Method (Zero-Shot)",
)
plot_cifar_band(
    ax2,
    pct_axis,
    wf_ft_acc_m,
    wf_ft_acc_s,
    "o-",
    "#d95f02",
    "Workflow Method (KD FT + EMA)",
)
plot_cifar_band(
    ax2,
    pct_axis,
    wf_zs_acc_m,
    wf_zs_acc_s,
    "x-.",
    "#e7298a",
    "Workflow Method (Zero-Shot)",
)
plot_cifar_band(
    ax2, pct_axis, da_ft_acc_m, da_ft_acc_s, "^-", "#6a3d9a", "DeepAbstract (KD FT)"
)
plot_cifar_band(
    ax2,
    pct_axis,
    da_zs_acc_m,
    da_zs_acc_s,
    "v-.",
    "#9467bd",
    "DeepAbstract (Zero-Shot)",
)

# Original model baseline (no perturbations)
ax2.axhline(
    y=teacher_acc_m,
    color="#d62728",
    linestyle="--",
    linewidth=2.0,
    label=f"Original Teacher ({teacher_acc_m:.2f}%)",
)
ax2.fill_between(
    [pct_axis[0], pct_axis[-1]],
    teacher_acc_m - teacher_acc_s,
    teacher_acc_m + teacher_acc_s,
    color="#d62728",
    alpha=0.15,
)

ax2.set_xlabel("Percentage of Conv Channels Kept (%)")
ax2.set_ylabel("Test Accuracy (%)")
ax2.set_title("CIFAR-10 Test Accuracy vs. % Channels Kept")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(loc="lower right", fontsize=8)

plt.tight_layout()
plt.savefig("cifar10_benchmark_multiseed.png", dpi=300)
plt.close()

print(
    f"{get_elapsed_str()} Benchmark complete. Output plot saved to 'cifar10_benchmark_multiseed.png'"
)
