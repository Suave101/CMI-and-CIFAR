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
SEEDS = [42]
TOTAL_CHANNELS = 64 + 128 + 256  # 448 Total Conv Channels
target_budgets = [45, 90, 135, 180, 225, 270, 315, 360]  # Kept channel budgets
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GLOBAL_START_TIME = time.time()

CKPT_DIR = "./checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


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
# 1. Vision Architecture & Modules
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


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, reduced, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(reduced, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = x.mean(dim=(2, 3))
        s = self.sigmoid(self.fc2(self.relu(self.fc1(z))))
        return x * s.unsqueeze(-1).unsqueeze(-1), s


class SECIFARConvNet(nn.Module):
    def __init__(self, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.se1 = SEBlock(64, reduction=reduction)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.se2 = SEBlock(128, reduction=reduction)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.se3 = SEBlock(256, reduction=reduction)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        h1, s1 = self.se1(self.relu1(self.conv1(x)))
        h1 = self.pool1(h1)
        h2, s2 = self.se2(self.relu2(self.conv2(h1)))
        h2 = self.pool2(h2)
        h3, s3 = self.se3(self.relu3(self.conv3(h2)))
        h3 = self.pool3(h3)
        out = self.fc(torch.flatten(h3, 1))
        return out, {"se1": s1, "se2": s2, "se3": s3}


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


class SEAbstractSubConvNet(nn.Module):
    def __init__(self, w1, b1, w2, b2, w3, b3, w_fc, b_fc, reduction=16):
        super().__init__()
        c1_out, c2_out, c3_out = w1.shape[0], w2.shape[0], w3.shape[0]

        self.conv1 = nn.Conv2d(3, c1_out, kernel_size=3, padding=1)
        self.se1 = SEBlock(c1_out, reduction=reduction)
        self.conv2 = nn.Conv2d(w2.shape[1], c2_out, kernel_size=3, padding=1)
        self.se2 = SEBlock(c2_out, reduction=reduction)
        self.conv3 = nn.Conv2d(w3.shape[1], c3_out, kernel_size=3, padding=1)
        self.se3 = SEBlock(c3_out, reduction=reduction)
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
        h1, s1 = self.se1(F.relu(self.conv1(x)))
        h1 = F.max_pool2d(h1, 2, 2)
        h2, s2 = self.se2(F.relu(self.conv2(h1)))
        h2 = F.max_pool2d(h2, 2, 2)
        h3, s3 = self.se3(F.relu(self.conv3(h2)))
        h3 = F.adaptive_avg_pool2d(h3, (1, 1))
        out = self.fc(torch.flatten(h3, 1))
        return out, {"se1": s1, "se2": s2, "se3": s3}


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
# 2. Helpers: Channel Medoid & Weight Caching & Distillation Loss
# =====================================================================
def get_or_train_teacher(teacher_model, seed, train_loader, epochs=15, device=device):
    ckpt_path = os.path.join(CKPT_DIR, f"teacher_seed_{seed}_latest.pth")
    if os.path.exists(ckpt_path):
        print(f"{get_elapsed_str()}   [Cache Hit] Loading Teacher Checkpoint ({ckpt_path})...")
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        teacher_model.load_state_dict(state_dict)
    else:
        print(f"{get_elapsed_str()}   [Cache Miss] Training Teacher Baseline ({epochs} Epochs)...")
        optimizer = optim.Adam(teacher_model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        teacher_model.train()
        for epoch in range(epochs):
            for x_b, y_b in train_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                loss = criterion(teacher_model(x_b), y_b)
                loss.backward()
                optimizer.step()
        torch.save({"state_dict": teacher_model.state_dict()}, ckpt_path)
        print(f"{get_elapsed_str()}   [Cache Saved] Saved Teacher Checkpoint to ({ckpt_path}).")
    teacher_model.eval()
    return teacher_model


def get_or_train_se_teacher(se_teacher, seed, train_loader, epochs=15, device=device):
    ckpt_path = os.path.join(CKPT_DIR, f"se_teacher_seed_{seed}_latest.pth")
    if os.path.exists(ckpt_path):
        print(f"{get_elapsed_str()}   [Cache Hit] Loading S&E Teacher Checkpoint ({ckpt_path})...")
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        se_teacher.load_state_dict(state_dict)
    else:
        print(f"{get_elapsed_str()}   [Cache Miss] Training S&E Teacher Baseline ({epochs} Epochs)...")
        optimizer = optim.Adam(se_teacher.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        se_teacher.train()
        for epoch in range(epochs):
            for x_b, y_b in train_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                out, _ = se_teacher(x_b)
                loss = criterion(out, y_b)
                loss.backward()
                optimizer.step()
        torch.save({"state_dict": se_teacher.state_dict()}, ckpt_path)
        print(f"{get_elapsed_str()}   [Cache Saved] Saved S&E Teacher Checkpoint to ({ckpt_path}).")
    se_teacher.eval()
    return se_teacher


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
# 3. Global Restructuring Workflow Method
# =====================================================================
def run_global_restructuring_workflow(
    teacher_model,
    val_loader,
    top_n_pct=15.0,
    n_clusters=3,  # 1 Core Layer + 3 Clusters = 4 Total Restructured Layers
    prune_pct_per_layer=20.0,
    device=device,
):
    teacher_model.eval()
    val_imgs = torch.cat([x.to(device) for x, _ in val_loader])

    # Step 1 & Step 2: Behavioral Fingerprinting & Causal Scoring
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

    # 3: Keep top n% of neurons (Layer 1 Core)
    total_neurons = len(neuron_map)
    n_top = int(np.ceil((top_n_pct / 100.0) * total_neurons))
    sorted_indices = np.argsort(ablation_scores)[::-1]
    top_n_indices = sorted_indices[:n_top]
    rem_indices = sorted_indices[n_top:]

    # 4: Agglomerative Clustering of (100-n)% of neurons (Layers 2, 3, 4)
    rem_fps = all_fps_safe[rem_indices]
    actual_clusters = min(n_clusters, len(rem_indices))
    clustering = AgglomerativeClustering(
        n_clusters=actual_clusters, metric="cosine", linkage="average"
    )
    cluster_labels = clustering.fit_predict(rem_fps)

    # 5: Global Restructuring into 4 distinct layers
    restructured_layers = {1: top_n_indices.tolist()}
    for c_id in range(actual_clusters):
        cluster_members = rem_indices[cluster_labels == c_id].tolist()
        restructured_layers[c_id + 2] = cluster_members

    # 6: Remove least important neurons per layer
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

            kept1_valid = kept1 if len(kept1) > 0 else [0]
            kept2_valid = kept2 if len(kept2) > 0 else [0]
            kept3_valid = kept3 if len(kept3) > 0 else [0]

            w1 = teacher_model.conv1.weight.detach()[kept1_valid, :, :, :]
            b1 = teacher_model.conv1.bias.detach()[kept1_valid]

            w2 = teacher_model.conv2.weight.detach()[kept2_valid, :, :, :][:, kept1_valid, :, :]
            b2 = teacher_model.conv2.bias.detach()[kept2_valid]

            w3 = teacher_model.conv3.weight.detach()[kept3_valid, :, :, :][:, kept2_valid, :, :]
            b3 = teacher_model.conv3.bias.detach()[kept3_valid]

            w_fc = teacher_model.fc.weight.detach()[:, kept3_valid]
            b_fc = teacher_model.fc.bias.detach()

            self.conv1 = nn.Conv2d(3, len(kept1_valid), kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(len(kept1_valid), len(kept2_valid), kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(len(kept2_valid), len(kept3_valid), kernel_size=3, padding=1)
            self.fc = nn.Linear(len(kept3_valid), 10)

            self.conv1.weight, self.conv1.bias = nn.Parameter(w1.clone()), nn.Parameter(b1.clone())
            self.conv2.weight, self.conv2.bias = nn.Parameter(w2.clone()), nn.Parameter(b2.clone())
            self.conv3.weight, self.conv3.bias = nn.Parameter(w3.clone()), nn.Parameter(b3.clone())
            self.fc.weight, self.fc.bias = nn.Parameter(w_fc.clone()), nn.Parameter(b_fc.clone())

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
# 4. Attention Visualizer (Outputs 5 Files of Attention by Layer)
# =====================================================================
def visualize_global_restructuring_se_attention(
    se_teacher, test_dataset, restructured_layers, save_dir="."
):
    """
    Evaluates Squeeze & Excitation channel attention across the 4 Global 
    Restructured Layers and saves 5 distinct visual figure files:
    - 4 Layer-by-Layer Spatial Attention Heatmap Overlay Figures on images by class.
    - 1 Overview Summary Matrix Figure.
    """
    se_teacher.eval()

    # 1. Collect 1 sample image per class from test set
    class_samples = {}
    for img, label in test_dataset:
        if label not in class_samples:
            class_samples[label] = img
        if len(class_samples) == 10:
            break

    images_tensor = torch.stack([class_samples[i] for i in range(10)]).to(device)

    # 2. Forward pass to collect unpooled intermediate spatial feature maps & SE channel scores
    with torch.no_grad():
        x1 = se_teacher.relu1(se_teacher.conv1(images_tensor))  # (10, 64, 32, 32)
        h1, s1 = se_teacher.se1(x1)                             # h1: (10, 64, 32, 32), s1: (10, 64)
        p1 = se_teacher.pool1(h1)                               # (10, 64, 16, 16)

        x2 = se_teacher.relu2(se_teacher.conv2(p1))             # (10, 128, 16, 16)
        h2, s2 = se_teacher.se2(x2)                             # h2: (10, 128, 16, 16), s2: (10, 128)
        p2 = se_teacher.pool2(h2)                               # (10, 128, 8, 8)

        x3 = se_teacher.relu3(se_teacher.conv3(p2))             # (10, 256, 8, 8)
        h3, s3 = se_teacher.se3(x3)                             # h3: (10, 256, 8, 8), s3: (10, 256)

    # 3. Generate Files 1-4: Class-wise Spatial Attention Maps Layer by Layer
    layer_names = [
        "Layer 1 (Core Top 15% Neurons)",
        "Layer 2 (Auxiliary Cluster 1)",
        "Layer 3 (Auxiliary Cluster 2)",
        "Layer 4 (Auxiliary Cluster 3)",
    ]
    file_names = [
        "cifar10_attention_layer1_core.png",
        "cifar10_attention_layer2_cluster1.png",
        "cifar10_attention_layer3_cluster2.png",
        "cifar10_attention_layer4_cluster3.png",
    ]

    for layer_id in range(1, 5):
        indices = restructured_layers.get(layer_id, [])
        layer_title = layer_names[layer_id - 1]
        out_path = os.path.join(save_dir, file_names[layer_id - 1])

        fig, axes = plt.subplots(2, 10, figsize=(20, 5))
        fig.suptitle(
            f"Spatial Channel Attention Overlay across Classes: {layer_title}",
            fontsize=15,
            fontweight="bold",
            y=0.98,
        )

        for c_idx in range(10):
            # Top Row: Clean RGB Sample Image
            img_np = class_samples[c_idx].numpy().transpose(1, 2, 0)
            img_np = np.clip(img_np, 0.0, 1.0)

            axes[0, c_idx].imshow(img_np)
            axes[0, c_idx].set_title(
                f"{CIFAR10_CLASSES[c_idx].capitalize()}",
                fontsize=11,
                fontweight="bold",
            )
            axes[0, c_idx].axis("off")

            # Compute channel-weighted spatial attention map
            spatial_map = torch.zeros((32, 32), device=device)
            if len(indices) > 0:
                for idx in indices:
                    if idx < 64:
                        feat = h1[c_idx, idx]
                        w = s1[c_idx, idx]
                        spatial_map += w * torch.abs(feat)
                    elif 64 <= idx < 192:
                        feat = h2[c_idx, idx - 64].unsqueeze(0).unsqueeze(0)
                        feat_up = F.interpolate(
                            feat, size=(32, 32), mode="bilinear", align_corners=False
                        ).squeeze()
                        w = s2[c_idx, idx - 64]
                        spatial_map += w * torch.abs(feat_up)
                    else:
                        feat = h3[c_idx, idx - 192].unsqueeze(0).unsqueeze(0)
                        feat_up = F.interpolate(
                            feat, size=(32, 32), mode="bilinear", align_corners=False
                        ).squeeze()
                        w = s3[c_idx, idx - 192]
                        spatial_map += w * torch.abs(feat_up)

            s_np = spatial_map.cpu().numpy()
            s_min, s_max = s_np.min(), s_np.max()
            s_norm = (s_np - s_min) / (s_max - s_min) if s_max - s_min > 1e-6 else np.zeros_like(s_np)

            # Bottom Row: Image Overlaid with Attention Heatmap
            axes[1, c_idx].imshow(img_np)
            axes[1, c_idx].imshow(s_norm, cmap="jet", alpha=0.55)
            axes[1, c_idx].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"{get_elapsed_str()} Saved Attention Figure [Layer {layer_id}/4]: '{out_path}'")

    # 4. Generate File 5: Layer-wise Summary Matrix Figure
    summary_fn = os.path.join(save_dir, "cifar10_attention_summary_matrix.png")
    legacy_fn = os.path.join(save_dir, "cifar10_global_restructuring_se_attention.png")

    se_all = torch.cat([s1, s2, s3], dim=1).cpu().numpy()  # (10, 448)
    layer_attentions = np.zeros((4, 10))

    for layer_id in range(1, 5):
        indices = restructured_layers.get(layer_id, [])
        if len(indices) > 0:
            layer_attentions[layer_id - 1, :] = se_all[:, indices].mean(axis=1)

    layer_attentions_norm = np.zeros_like(layer_attentions)
    for i in range(4):
        row_min, row_max = layer_attentions[i].min(), layer_attentions[i].max()
        if row_max - row_min > 1e-6:
            layer_attentions_norm[i] = (layer_attentions[i] - row_min) / (row_max - row_min)
        else:
            layer_attentions_norm[i] = 0.5

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 10, height_ratios=[1, 2.2], hspace=0.35, wspace=0.2)

    for i in range(10):
        ax = fig.add_subplot(gs[0, i])
        img_np = class_samples[i].numpy().transpose(1, 2, 0)
        img_np = np.clip(img_np, 0.0, 1.0)
        ax.imshow(img_np)
        ax.set_title(f"{CIFAR10_CLASSES[i].capitalize()}", fontsize=11, fontweight="bold")
        ax.axis("off")

    ax_heatmap = fig.add_subplot(gs[1, :])
    im = ax_heatmap.imshow(layer_attentions_norm, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)

    y_labels = [
        "Layer 1 (Core Top 15%)",
        "Layer 2 (Aux Cluster 1)",
        "Layer 3 (Aux Cluster 2)",
        "Layer 4 (Aux Cluster 3)",
    ]
    ax_heatmap.set_yticks(np.arange(4))
    ax_heatmap.set_yticklabels(y_labels, fontsize=11, fontweight="bold")
    ax_heatmap.set_xticks(np.arange(10))
    ax_heatmap.set_xticklabels([c.capitalize() for c in CIFAR10_CLASSES], fontsize=11, fontweight="bold")
    ax_heatmap.set_title(
        "Relative Squeeze & Excitation Channel Attention Contrast per Restructured Layer",
        fontsize=13,
        pad=12,
        fontweight="bold",
    )

    for i in range(4):
        for j in range(10):
            raw_val = layer_attentions[i, j]
            norm_val = layer_attentions_norm[i, j]
            color = "white" if norm_val > 0.65 else "black"
            ax_heatmap.text(
                j, i, f"Abs: {raw_val:.3f}\nRel: {norm_val:.2f}",
                ha="center", va="center", color=color, fontsize=9, fontweight="bold"
            )

    cbar = fig.colorbar(im, ax=ax_heatmap, orientation="horizontal", pad=0.18, shrink=0.6)
    cbar.set_label("Normalized Relative Attention Score (Min-Max per Layer)", fontsize=10)

    plt.savefig(summary_fn, dpi=300, bbox_inches="tight")
    plt.savefig(legacy_fn, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"{get_elapsed_str()} Saved Attention Figure [Summary Matrix]: '{summary_fn}' and '{legacy_fn}'")


# =====================================================================
# 5. Parallel Squeeze-and-Excitation Benchmark Evaluator
# =====================================================================
def run_cifar_se_benchmark(seed, train_loader, val_loader, test_loader, X_test, y_test):
    print(f"{get_elapsed_str()}   [S&E Pipeline] Preparing S&E-Enhanced ConvNet...")
    se_teacher = SECIFARConvNet().to(device)
    se_teacher = get_or_train_se_teacher(se_teacher, seed, train_loader, epochs=15, device=device)

    se_teacher.eval()
    with torch.no_grad():
        se_teacher_logits, _ = se_teacher(X_test)
        se_teacher_acc = (
            se_teacher_logits.argmax(dim=1) == y_test
        ).float().mean().item() * 100.0
        se_orig_preds = se_teacher_logits.argmax(dim=1)

    print(f"{get_elapsed_str()}   [S&E Teacher Accuracy]: {se_teacher_acc:.2f}%")

    val_imgs = torch.cat([x.to(device) for x, _ in val_loader])
    with torch.no_grad():
        val_teacher_logits, _ = se_teacher(val_imgs)
        c1_act, _ = se_teacher.se1(se_teacher.relu1(se_teacher.conv1(val_imgs)))
        c1_act = se_teacher.pool1(c1_act)
        c2_act, _ = se_teacher.se2(se_teacher.relu2(se_teacher.conv2(c1_act)))
        c2_act = se_teacher.pool2(c2_act)
        c3_act, _ = se_teacher.se3(se_teacher.relu3(se_teacher.conv3(c2_act)))
        c3_act = se_teacher.pool3(c3_act)

    fp_c1 = c1_act.mean(dim=(2, 3)).T.cpu().numpy()
    fp_c2 = c2_act.mean(dim=(2, 3)).T.cpu().numpy()
    fp_c3 = c3_act.mean(dim=(2, 3)).T.cpu().numpy()

    fp_c1_safe, fp_c2_safe, fp_c3_safe = (
        make_safe_fp(fp_c1),
        make_safe_fp(fp_c2),
        make_safe_fp(fp_c3),
    )
    mean_act_c1, mean_act_c2, mean_act_c3 = (
        fp_c1.mean(axis=1),
        fp_c2.mean(axis=1),
        fp_c3.mean(axis=1),
    )

    scores_c1, scores_c2, scores_c3 = np.zeros(64), np.zeros(128), np.zeros(256)

    with torch.no_grad():
        for i in range(64):
            temp_c1 = c1_act.clone()
            temp_c1[:, i, :, :] = 0.0
            h2, _ = se_teacher.se2(se_teacher.relu2(se_teacher.conv2(temp_c1)))
            h2 = se_teacher.pool2(h2)
            h3, _ = se_teacher.se3(se_teacher.relu3(se_teacher.conv3(h2)))
            h3 = se_teacher.pool3(h3)
            out = se_teacher.fc(torch.flatten(h3, 1))
            scores_c1[i] = (
                (1.0 - F.cosine_similarity(val_teacher_logits, out, dim=1))
                .mean()
                .item()
            )

        for j in range(128):
            temp_c2 = c2_act.clone()
            temp_c2[:, j, :, :] = 0.0
            h3, _ = se_teacher.se3(se_teacher.relu3(se_teacher.conv3(temp_c2)))
            h3 = se_teacher.pool3(h3)
            out = se_teacher.fc(torch.flatten(h3, 1))
            scores_c2[j] = (
                (1.0 - F.cosine_similarity(val_teacher_logits, out, dim=1))
                .mean()
                .item()
            )

        for k in range(256):
            temp_c3 = c3_act.clone()
            temp_c3[:, k, :, :] = 0.0
            out = se_teacher.fc(torch.flatten(temp_c3, 1))
            scores_c3[k] = (
                (1.0 - F.cosine_similarity(val_teacher_logits, out, dim=1))
                .mean()
                .item()
            )

    se_res = {
        "se_zs_acc": [],
        "se_zs_agr": [],
        "se_ft_acc": [],
        "se_ft_agr": [],
        "pct_kept": [],
        "se_teacher_acc": se_teacher_acc,
    }

    for target_channels in target_budgets:
        b1 = max(1, int(round(target_channels * (64 / 448))))
        b2 = max(1, int(round(target_channels * (128 / 448))))
        b3 = max(1, target_channels - b1 - b2)

        res_ratio = 0.10 if target_channels <= 150 else 0.15
        n_top1, n_top2, n_top3 = (
            int(b1 * res_ratio),
            int(b2 * res_ratio),
            int(b3 * res_ratio),
        )
        k_c1, k_c2, k_c3 = max(1, b1 - n_top1), max(1, b2 - n_top2), max(1, b3 - n_top3)

        top1 = set(np.argsort(scores_c1)[-n_top1:]) if n_top1 > 0 else set()
        top2 = set(np.argsort(scores_c2)[-n_top2:]) if n_top2 > 0 else set()
        top3 = set(np.argsort(scores_c3)[-n_top3:]) if n_top3 > 0 else set()

        rem1 = np.array([i for i in range(64) if i not in top1])
        rem2 = np.array([j for j in range(128) if j not in top2])
        rem3 = np.array([k for k in range(256) if k not in top3])

        W1, b1_t = (
            se_teacher.conv1.weight.detach().clone(),
            se_teacher.conv1.bias.detach().clone(),
        )
        W2, b2_t = (
            se_teacher.conv2.weight.detach().clone(),
            se_teacher.conv2.bias.detach().clone(),
        )
        W3, b3_t = (
            se_teacher.conv3.weight.detach().clone(),
            se_teacher.conv3.bias.detach().clone(),
        )
        W_fc, b_fc = (
            se_teacher.fc.weight.detach().clone(),
            se_teacher.fc.bias.detach().clone(),
        )

        kept1, kept2, kept3 = set(top1), set(top2), set(top3)

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

        m1, m2, m3 = (
            np.array([i in kept1 for i in range(64)]),
            np.array([j in kept2 for j in range(128)]),
            np.array([k in kept3 for k in range(256)]),
        )

        sub_w1, sub_b1 = W1[m1, :, :, :], b1_t[m1]
        sub_w2, sub_b2 = W2[m2, :, :, :][:, m1, :, :], b2_t[m2]
        sub_w3, sub_b3 = W3[m3, :, :, :][:, m2, :, :], b3_t[m3]
        sub_wfc, sub_bfc = W_fc[:, m3], b_fc

        se_sub_net = SEAbstractSubConvNet(
            sub_w1, sub_b1, sub_w2, sub_b2, sub_w3, sub_b3, sub_wfc, sub_bfc
        ).to(device)

        se_sub_net.eval()
        with torch.no_grad():
            zs_logits, _ = se_sub_net(X_test)
            zs_preds = zs_logits.argmax(dim=1)
            zs_acc = (zs_preds == y_test).float().mean().item() * 100.0
            zs_agr = (zs_preds == se_orig_preds).float().mean().item() * 100.0

        opt = optim.Adam(se_sub_net.parameters(), lr=1e-3)
        ema = EMAModel(se_sub_net, decay=0.98)
        se_sub_net.train()

        for epoch in range(5):
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                with torch.no_grad():
                    t_logits, _ = se_teacher(bx)
                opt.zero_grad()
                s_logits, _ = se_sub_net(bx)
                loss = kd_loss_fn(s_logits, t_logits, by)
                loss.backward()
                opt.step()
                ema.update()

        ema.apply_shadow()
        se_sub_net.eval()
        with torch.no_grad():
            ft_logits, _ = se_sub_net(X_test)
            ft_preds = ft_logits.argmax(dim=1)
            ft_acc = (ft_preds == y_test).float().mean().item() * 100.0
            ft_agr = (ft_preds == se_orig_preds).float().mean().item() * 100.0
        ema.restore()

        total_k = len(kept1) + len(kept2) + len(kept3)
        pct = (total_k / TOTAL_CHANNELS) * 100.0

        se_res["pct_kept"].append(pct)
        se_res["se_zs_acc"].append(zs_acc)
        se_res["se_zs_agr"].append(zs_agr)
        se_res["se_ft_acc"].append(ft_acc)
        se_res["se_ft_agr"].append(ft_agr)

    return se_res, se_teacher


# =====================================================================
# 6. Single-Seed Benchmark Execution
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

    data_root = "/shared/datasets/cifar-10" if os.path.exists("/shared/datasets/cifar-10") else "./data"
    download_flag = not os.path.exists(data_root)

    full_train = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        download=download_flag,
        transform=transform_train,
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        download=download_flag,
        transform=transform_test,
    )

    targets = np.array(full_train.targets)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))

    train_loader = DataLoader(
        Subset(full_train, train_idx), batch_size=128, shuffle=True
    )
    val_loader = DataLoader(Subset(full_train, val_idx), batch_size=128, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    teacher_model = CIFARConvNet().to(device)
    teacher_model = get_or_train_teacher(teacher_model, seed, train_loader, epochs=15, device=device)

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

    def build_deepabstract_core(target_channels):
        t0_zs = time.time()
        b1 = max(1, int(round(target_channels * (64 / 448))))
        b2 = max(1, int(round(target_channels * (128 / 448))))
        b3 = max(1, target_channels - b1 - b2)

        val_imgs = torch.cat([x.to(device) for x, _ in val_loader])
        with torch.no_grad():
            c1_act = teacher_model.pool1(
                teacher_model.relu1(teacher_model.conv1(val_imgs))
            )
            c2_act = teacher_model.pool2(
                teacher_model.relu2(teacher_model.conv2(c1_act))
            )
            c3_act = teacher_model.pool3(
                teacher_model.relu3(teacher_model.conv3(c2_act))
            )

        fp_c1 = c1_act.mean(dim=(2, 3)).T.cpu().numpy()
        fp_c2 = c2_act.mean(dim=(2, 3)).T.cpu().numpy()
        fp_c3 = c3_act.mean(dim=(2, 3)).T.cpu().numpy()

        fp_c1_s, fp_c2_s, fp_c3_s = (
            make_safe_fp(fp_c1),
            make_safe_fp(fp_c2),
            make_safe_fp(fp_c3),
        )
        mean_c1, mean_c2, mean_c3 = (
            fp_c1.mean(axis=1),
            fp_c2.mean(axis=1),
            fp_c3.mean(axis=1),
        )

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

        kept1, kept2, kept3 = set(), set(), set()

        if len(fp_c1_s) > 0 and b1 > 0:
            c1 = AgglomerativeClustering(
                n_clusters=min(b1, len(fp_c1_s)), metric="cosine", linkage="average"
            )
            labels1 = c1.fit_predict(fp_c1_s)
            for cid in range(b1):
                m = np.where(labels1 == cid)[0]
                if len(m) == 0:
                    continue
                rep = find_channel_medoid(fp_c1_s, mean_c1, m)
                kept1.add(rep)
                W1[rep] = W1[m].mean(dim=0)
                b1_t[rep] = b1_t[m].mean(dim=0)

        if len(fp_c2_s) > 0 and b2 > 0:
            c2 = AgglomerativeClustering(
                n_clusters=min(b2, len(fp_c2_s)), metric="cosine", linkage="average"
            )
            labels2 = c2.fit_predict(fp_c2_s)
            for cid in range(b2):
                m = np.where(labels2 == cid)[0]
                if len(m) == 0:
                    continue
                rep = find_channel_medoid(fp_c2_s, mean_c2, m)
                kept2.add(rep)
                W2[rep] = W2[m].mean(dim=0)
                b2_t[rep] = b2_t[m].mean(dim=0)

        if len(fp_c3_s) > 0 and b3 > 0:
            c3 = AgglomerativeClustering(
                n_clusters=min(b3, len(fp_c3_s)), metric="cosine", linkage="average"
            )
            labels3 = c3.fit_predict(fp_c3_s)
            for cid in range(b3):
                m = np.where(labels3 == cid)[0]
                if len(m) == 0:
                    continue
                rep = find_channel_medoid(fp_c3_s, mean_c3, m)
                kept3.add(rep)
                W3[rep] = W3[m].mean(dim=0)
                b3_t[rep] = b3_t[m].mean(dim=0)

        m1 = np.array([i in kept1 for i in range(64)])
        m2 = np.array([j in kept2 for j in range(128)])
        m3 = np.array([k in kept3 for k in range(256)])

        sub_net = AbstractSubConvNet(
            W1[m1, :, :, :],
            b1_t[m1],
            W2[m2, :, :, :][:, m1, :, :],
            b2_t[m2],
            W3[m3, :, :, :][:, m2, :, :],
            b3_t[m3],
            W_fc[:, m3],
            b_fc,
        ).to(device)

        sub_net.eval()
        with torch.no_grad():
            zs_preds = sub_net(X_test).argmax(dim=1)
            zs_acc = (zs_preds == y_test).float().mean().item() * 100.0
            zs_agr = (zs_preds == orig_test_preds).float().mean().item() * 100.0
        t_zs = (time.time() - t0_zs) * 1000.0

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
                loss = kd_loss_fn(sub_net(bx), t_logits, by)
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
        "da_zs_acc": [],
        "da_zs_agr": [],
        "da_ft_acc": [],
        "da_ft_agr": [],
        "gr_zs_acc": [],
        "gr_zs_agr": [],
        "gr_ft_acc": [],
        "gr_ft_agr": [],
        "pct_kept": [],
        "da_zs_t": [],
        "da_ft_t": [],
        "gr_zs_t": [],
        "gr_ft_t": [],
        "teacher_acc": teacher_acc,
    }

    last_gr_out = None

    for b in target_budgets:
        t_b_start = time.time()
        d_za, d_zg, d_fa, d_fg, total_k, d_tz, d_tf = build_deepabstract_core(b)

        prune_pct = max(0.0, min(99.0, (1.0 - b / TOTAL_CHANNELS) * 100.0))
        gr_out = run_global_restructuring_workflow(
            teacher_model,
            val_loader,
            top_n_pct=15.0,
            n_clusters=3,
            prune_pct_per_layer=prune_pct,
            device=device,
        )
        last_gr_out = gr_out
        gr_net = gr_out["restructured_model"]

        gr_net.eval()
        t0_gr_zs = time.time()
        with torch.no_grad():
            gr_zs_logits = gr_net(X_test)
            gr_zs_preds = gr_zs_logits.argmax(dim=1)
            gr_zs_acc = (gr_zs_preds == y_test).float().mean().item() * 100.0
            gr_zs_agr = (gr_zs_preds == orig_test_preds).float().mean().item() * 100.0
        gr_tz = (time.time() - t0_gr_zs) * 1000.0

        t0_gr_ft = time.time()
        gr_opt = optim.Adam(gr_net.parameters(), lr=1e-3)
        gr_ema = EMAModel(gr_net, decay=0.98)
        gr_net.train()
        for epoch in range(5):
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                with torch.no_grad():
                    t_logits = teacher_model(bx)
                gr_opt.zero_grad()
                loss = kd_loss_fn(gr_net(bx), t_logits, by)
                loss.backward()
                gr_opt.step()
                gr_ema.update()

        gr_ema.apply_shadow()
        gr_net.eval()
        with torch.no_grad():
            gr_ft_preds = gr_net(X_test).argmax(dim=1)
            gr_ft_acc = (gr_ft_preds == y_test).float().mean().item() * 100.0
            gr_ft_agr = (gr_ft_preds == orig_test_preds).float().mean().item() * 100.0
        gr_ema.restore()
        gr_tf = (time.time() - t0_gr_ft) * 1000.0

        pct = (total_k / TOTAL_CHANNELS) * 100.0
        res["pct_kept"].append(pct)
        res["da_zs_acc"].append(d_za)
        res["da_zs_agr"].append(d_zg)
        res["da_ft_acc"].append(d_fa)
        res["da_ft_agr"].append(d_fg)
        res["gr_zs_acc"].append(gr_zs_acc)
        res["gr_zs_agr"].append(gr_zs_agr)
        res["gr_ft_acc"].append(gr_ft_acc)
        res["gr_ft_agr"].append(gr_ft_agr)
        res["da_zs_t"].append(d_tz)
        res["da_ft_t"].append(d_tf)
        res["gr_zs_t"].append(gr_tz)
        res["gr_ft_t"].append(gr_tf)

        b_time = time.time() - t_b_start
        print(
            f"{get_elapsed_str()}     -> Budget {pct:4.1f}% ({total_k:3d} ch) | GR FT: {gr_ft_acc:.2f}% | DA FT: {d_fa:.2f}% ({b_time:.1f}s)"
        )

    se_res, se_teacher = run_cifar_se_benchmark(
        seed, train_loader, val_loader, test_loader, X_test, y_test
    )

    test_dataset_raw = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        download=download_flag,
        transform=transforms.ToTensor(),
    )
    visualize_global_restructuring_se_attention(
        se_teacher, test_dataset_raw, last_gr_out["restructured_layers"]
    )

    return res, se_res


# =====================================================================
# 7. Main Benchmark Orchestrator & Plotting
# =====================================================================
print(f"=== STARTING CIFAR-10 BENCHMARK ({len(SEEDS)} SEEDS) ===")
all_cifar_runs = []
all_se_runs = []

for idx, s in enumerate(SEEDS):
    run_res, se_res = run_cifar_seed_benchmark(idx, s)
    all_cifar_runs.append(run_res)
    all_se_runs.append(se_res)

pct_axis = np.mean([r["pct_kept"] for r in all_cifar_runs], axis=0)


def aggregate_cifar(key):
    data = np.array([r[key] for r in all_cifar_runs])
    return np.mean(data, axis=0), np.std(data, axis=0)


def aggregate_se(key):
    data = np.array([r[key] for r in all_se_runs])
    return np.mean(data, axis=0), np.std(data, axis=0)


da_ft_acc_m, da_ft_acc_s = aggregate_cifar("da_ft_acc")
da_ft_agr_m, da_ft_agr_s = aggregate_cifar("da_ft_agr")
da_zs_acc_m, da_zs_acc_s = aggregate_cifar("da_zs_acc")
da_zs_agr_m, da_zs_agr_s = aggregate_cifar("da_zs_agr")

gr_ft_acc_m, gr_ft_acc_s = aggregate_cifar("gr_ft_acc")
gr_ft_agr_m, gr_ft_agr_s = aggregate_cifar("gr_ft_agr")
gr_zs_acc_m, gr_zs_acc_s = aggregate_cifar("gr_zs_acc")
gr_zs_agr_m, gr_zs_agr_s = aggregate_cifar("gr_zs_agr")

teacher_acc_m = np.mean([r["teacher_acc"] for r in all_cifar_runs])
teacher_acc_s = np.std([r["teacher_acc"] for r in all_cifar_runs])

se_ft_acc_m, se_ft_acc_s = aggregate_se("se_ft_acc")
se_ft_agr_m, se_ft_agr_s = aggregate_se("se_ft_agr")
se_zs_acc_m, se_zs_acc_s = aggregate_se("se_zs_acc")
se_zs_agr_m, se_zs_agr_s = aggregate_se("se_zs_agr")
se_teacher_acc_m = np.mean([r["se_teacher_acc"] for r in all_se_runs])
se_teacher_acc_s = np.std([r["se_teacher_acc"] for r in all_se_runs])

print(f"\n{get_elapsed_str()} Benchmark complete! Generated attention figures & checkpoints in project directory.")