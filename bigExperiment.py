import time
import random
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

# =====================================================================
# 0. Global Setup & Device Routing
# =====================================================================
SEEDS = [42, 101, 2024, 7, 99]
TOTAL_CHANNELS = 64 + 128 + 256 + 512  # 960 Total Stage Channels in ResNet-18
target_budgets = [96, 192, 288, 384, 480, 576, 672, 768]  # Kept channel budgets (10% to 80%)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
# 1. ResNet-18 Vision Architecture & Dynamic Sub-Network Module
# =====================================================================
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CIFARResNet18(nn.Module):
    def __init__(self, c1=64, c2=128, c3=256, c4=512, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(c1, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(c2, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(c3, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(c4, num_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c4, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        s1 = self.layer1(out)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        out = self.avgpool(s4)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

def create_sub_resnet(teacher, kept1, kept2, kept3, kept4):
    """Constructs a dynamically sized ResNet-18 and copies sliced/aggregated weights."""
    m1 = np.array(sorted(list(kept1)))
    m2 = np.array(sorted(list(kept2)))
    m3 = np.array(sorted(list(kept3)))
    m4 = np.array(sorted(list(kept4)))

    c1, c2, c3, c4 = len(m1), len(m2), len(m3), len(m4)
    sub_net = CIFARResNet18(c1=c1, c2=c2, c3=c3, c4=c4).to(device)

    # 1. Stem
    sub_net.conv1.weight.data.copy_(teacher.conv1.weight.data)
    sub_net.bn1.weight.data.copy_(teacher.bn1.weight.data)
    sub_net.bn1.bias.data.copy_(teacher.bn1.bias.data)
    sub_net.bn1.running_mean.data.copy_(teacher.bn1.running_mean.data)
    sub_net.bn1.running_var.data.copy_(teacher.bn1.running_var.data)

    def slice_basic_block(src, dst, in_m, out_m, is_first_block):
        prev_m = in_m if is_first_block else out_m
        dst.conv1.weight.data.copy_(src.conv1.weight.data[out_m][:, prev_m])
        dst.bn1.weight.data.copy_(src.bn1.weight.data[out_m])
        dst.bn1.bias.data.copy_(src.bn1.bias.data[out_m])
        dst.bn1.running_mean.data.copy_(src.bn1.running_mean.data[out_m])
        dst.bn1.running_var.data.copy_(src.bn1.running_var.data[out_m])

        dst.conv2.weight.data.copy_(src.conv2.weight.data[out_m][:, out_m])
        dst.bn2.weight.data.copy_(src.bn2.weight.data[out_m])
        dst.bn2.bias.data.copy_(src.bn2.bias.data[out_m])
        dst.bn2.running_mean.data.copy_(src.bn2.running_mean.data[out_m])
        dst.bn2.running_var.data.copy_(src.bn2.running_var.data[out_m])

        if len(src.shortcut) > 0:
            dst.shortcut[0].weight.data.copy_(src.shortcut[0].weight.data[out_m][:, prev_m])
            dst.shortcut[1].weight.data.copy_(src.shortcut[1].weight.data[out_m])
            dst.shortcut[1].bias.data.copy_(src.shortcut[1].bias.data[out_m])
            dst.shortcut[1].running_mean.data.copy_(src.shortcut[1].running_mean.data[out_m])
            dst.shortcut[1].running_var.data.copy_(src.shortcut[1].running_var.data[out_m])

    in_m_0 = np.arange(64)
    slice_basic_block(teacher.layer1[0], sub_net.layer1[0], in_m_0, m1, is_first_block=True)
    slice_basic_block(teacher.layer1[1], sub_net.layer1[1], m1, m1, is_first_block=False)

    slice_basic_block(teacher.layer2[0], sub_net.layer2[0], m1, m2, is_first_block=True)
    slice_basic_block(teacher.layer2[1], sub_net.layer2[1], m2, m2, is_first_block=False)

    slice_basic_block(teacher.layer3[0], sub_net.layer3[0], m2, m3, is_first_block=True)
    slice_basic_block(teacher.layer3[1], sub_net.layer3[1], m3, m3, is_first_block=False)

    slice_basic_block(teacher.layer4[0], sub_net.layer4[0], m3, m4, is_first_block=True)
    slice_basic_block(teacher.layer4[1], sub_net.layer4[1], m4, m4, is_first_block=False)

    sub_net.fc.weight.data.copy_(teacher.fc.weight.data[:, m4])
    sub_net.fc.bias.data.copy_(teacher.fc.bias.data)

    return sub_net

class EMAModel:
    def __init__(self, model, decay=0.98):
        self.model = model
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    def update(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = (1.0 - self.decay) * p.data + self.decay * self.shadow[n]

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
    if len(members) == 1: return members[0]
    sub_fps = fps[members]
    centroid = sub_fps.mean(axis=0)
    dists = np.array([1.0 - np.dot(fp, centroid)/(np.linalg.norm(fp)*np.linalg.norm(centroid)+1e-8) for fp in sub_fps])
    penalty = np.where(mean_acts[members] < 1e-5, 10.0, 0.0)
    return members[np.argmin(dists + penalty)]

def kd_loss_fn(s_logits, t_logits, labels, T=2.0, alpha=0.85):
    soft_t = F.softmax(t_logits / T, dim=1)
    soft_s = F.log_softmax(s_logits / T, dim=1)
    kl = F.kl_div(soft_s, soft_t, reduction='batchmean') * (T ** 2)
    ce = F.cross_entropy(s_logits, labels)
    return alpha * kl + (1.0 - alpha) * ce

# =====================================================================
# 3. Single-Seed Benchmark Execution
# =====================================================================
def run_cifar_seed_benchmark(seed_idx, seed):
    set_seed(seed)
    print(f"\n{get_elapsed_str()} === Starting Seed [{seed_idx + 1}/{len(SEEDS)}] (Seed: {seed}) ===")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    full_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    indices = list(range(len(full_train)))
    np.random.shuffle(indices)
    train_idx, val_idx = indices[:40000], indices[40000:]

    train_loader = DataLoader(Subset(full_train, train_idx), batch_size=128, shuffle=True)
    val_loader = DataLoader(Subset(full_train, val_idx), batch_size=128, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    teacher_model = CIFARResNet18().to(device)
    optimizer = optim.Adam(teacher_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    print(f"{get_elapsed_str()}   [Step 1/3] Training ResNet-18 Teacher Baseline (15 Epochs)...")
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
            print(f"{get_elapsed_str()}     -> Teacher Epoch {epoch+1:02d}/15 | Loss: {running_loss/len(train_loader):.4f}")
    
    teacher_model.eval()

    test_imgs, test_targets = [], []
    for x_b, y_b in test_loader:
        test_imgs.append(x_b)
        test_targets.append(y_b)
    X_test = torch.cat(test_imgs).to(device)
    y_test = torch.cat(test_targets).to(device)

    with torch.no_grad():
        teacher_acc = (teacher_model(X_test).argmax(dim=1) == y_test).float().mean().item() * 100.0
        orig_test_preds = teacher_model(X_test).argmax(dim=1)
    
    print(f"{get_elapsed_str()}   [ResNet-18 Teacher Accuracy]: {teacher_acc:.2f}%")

    # -----------------------------------------------------------------
    # Causal Channel Profiling on VALIDATION SET ONLY
    # -----------------------------------------------------------------
    print(f"{get_elapsed_str()}   [Step 2/3] Profiling Causal Stage Channels & Computing Fingerprints...")
    val_imgs = torch.cat([x.to(device) for x, _ in val_loader])
    with torch.no_grad():
        val_teacher_logits = teacher_model(val_imgs)
        x0 = F.relu(teacher_model.bn1(teacher_model.conv1(val_imgs)))
        s1_act = teacher_model.layer1(x0)
        s2_act = teacher_model.layer2(s1_act)
        s3_act = teacher_model.layer3(s2_act)
        s4_act = teacher_model.layer4(s3_act)

    fp_s1 = s1_act.mean(dim=(2, 3)).T.cpu().numpy()
    fp_s2 = s2_act.mean(dim=(2, 3)).T.cpu().numpy()
    fp_s3 = s3_act.mean(dim=(2, 3)).T.cpu().numpy()
    fp_s4 = s4_act.mean(dim=(2, 3)).T.cpu().numpy()

    fp_s1_safe = make_safe_fp(fp_s1)
    fp_s2_safe = make_safe_fp(fp_s2)
    fp_s3_safe = make_safe_fp(fp_s3)
    fp_s4_safe = make_safe_fp(fp_s4)

    mean_act_s1 = fp_s1.mean(axis=1)
    mean_act_s2 = fp_s2.mean(axis=1)
    mean_act_s3 = fp_s3.mean(axis=1)
    mean_act_s4 = fp_s4.mean(axis=1)

    scores_s1 = np.zeros(64)
    scores_s2 = np.zeros(128)
    scores_s3 = np.zeros(256)
    scores_s4 = np.zeros(512)

    with torch.no_grad():
        for i in range(64):
            temp_s1 = s1_act.clone(); temp_s1[:, i, :, :] = 0.0
            out = teacher_model.fc(torch.flatten(teacher_model.avgpool(teacher_model.layer4(teacher_model.layer3(teacher_model.layer2(temp_s1)))), 1))
            scores_s1[i] = (1.0 - F.cosine_similarity(val_teacher_logits, out, dim=1)).mean().item()

        for j in range(128):
            temp_s2 = s2_act.clone(); temp_s2[:, j, :, :] = 0.0
            out = teacher_model.fc(torch.flatten(teacher_model.avgpool(teacher_model.layer4(teacher_model.layer3(temp_s2))), 1))
            scores_s2[j] = (1.0 - F.cosine_similarity(val_teacher_logits, out, dim=1)).mean().item()

        for k in range(256):
            temp_s3 = s3_act.clone(); temp_s3[:, k, :, :] = 0.0
            out = teacher_model.fc(torch.flatten(teacher_model.avgpool(teacher_model.layer4(temp_s3)), 1))
            scores_s3[k] = (1.0 - F.cosine_similarity(val_teacher_logits, out, dim=1)).mean().item()

        for m in range(512):
            temp_s4 = s4_act.clone(); temp_s4[:, m, :, :] = 0.0
            out = teacher_model.fc(torch.flatten(teacher_model.avgpool(temp_s4), 1))
            scores_s4[m] = (1.0 - F.cosine_similarity(val_teacher_logits, out, dim=1)).mean().item()

    # -----------------------------------------------------------------
    # Benchmark Target Budgets
    # -----------------------------------------------------------------
    print(f"{get_elapsed_str()}   [Step 3/3] Running Abstraction Budgets...")

    def build_conv_core(target_channels, use_causal=True):
        t0_zs = time.time()
        b1 = max(1, int(round(target_channels * (64 / TOTAL_CHANNELS))))
        b2 = max(1, int(round(target_channels * (128 / TOTAL_CHANNELS))))
        b3 = max(1, int(round(target_channels * (256 / TOTAL_CHANNELS))))
        b4 = max(1, target_channels - b1 - b2 - b3)

        if use_causal:
            res_ratio = 0.10 if target_channels <= 300 else 0.15
            n_top1, n_top2 = int(b1 * res_ratio), int(b2 * res_ratio)
            n_top3, n_top4 = int(b3 * res_ratio), int(b4 * res_ratio)
            
            k_c1, k_c2 = max(1, b1 - n_top1), max(1, b2 - n_top2)
            k_c3, k_c4 = max(1, b3 - n_top3), max(1, b4 - n_top4)
            
            top1 = set(np.argsort(scores_s1)[-n_top1:]) if n_top1 > 0 else set()
            top2 = set(np.argsort(scores_s2)[-n_top2:]) if n_top2 > 0 else set()
            top3 = set(np.argsort(scores_s3)[-n_top3:]) if n_top3 > 0 else set()
            top4 = set(np.argsort(scores_s4)[-n_top4:]) if n_top4 > 0 else set()
        else:
            k_c1, k_c2, k_c3, k_c4 = b1, b2, b3, b4
            top1, top2, top3, top4 = set(), set(), set(), set()

        rem1 = np.array([i for i in range(64) if i not in top1])
        rem2 = np.array([j for j in range(128) if j not in top2])
        rem3 = np.array([k for k in range(256) if k not in top3])
        rem4 = np.array([m for m in range(512) if m not in top4])

        t_model = copy.deepcopy(teacher_model)
        kept1, kept2, kept3, kept4 = set(top1), set(top2), set(top3), set(top4)

        # Stage 1 Clustering
        if len(rem1) > 0 and k_c1 > 0:
            c1 = AgglomerativeClustering(n_clusters=min(k_c1, len(rem1)), metric='cosine', linkage='average')
            labels1 = c1.fit_predict(fp_s1_safe[rem1])
            for cid in range(k_c1):
                m = rem1[labels1 == cid]
                if len(m) == 0: continue
                rep = find_channel_medoid(fp_s1_safe, mean_act_s1, m)
                kept1.add(rep)
                t_model.layer1[1].conv2.weight.data[rep] = t_model.layer1[1].conv2.weight.data[m].mean(dim=0)
                t_model.layer1[1].bn2.weight.data[rep] = t_model.layer1[1].bn2.weight.data[m].mean(dim=0)
                t_model.layer1[1].bn2.bias.data[rep] = t_model.layer1[1].bn2.bias.data[m].mean(dim=0)
                norm = np.sqrt(len(m))
                for x in m:
                    if x != rep:
                        t_model.layer2[0].conv1.weight.data[:, rep] += t_model.layer2[0].conv1.weight.data[:, x] / norm
                        if len(t_model.layer2[0].shortcut) > 0:
                            t_model.layer2[0].shortcut[0].weight.data[:, rep] += t_model.layer2[0].shortcut[0].weight.data[:, x] / norm

        # Stage 2 Clustering
        if len(rem2) > 0 and k_c2 > 0:
            c2 = AgglomerativeClustering(n_clusters=min(k_c2, len(rem2)), metric='cosine', linkage='average')
            labels2 = c2.fit_predict(fp_s2_safe[rem2])
            for cid in range(k_c2):
                m = rem2[labels2 == cid]
                if len(m) == 0: continue
                rep = find_channel_medoid(fp_s2_safe, mean_act_s2, m)
                kept2.add(rep)
                t_model.layer2[1].conv2.weight.data[rep] = t_model.layer2[1].conv2.weight.data[m].mean(dim=0)
                t_model.layer2[1].bn2.weight.data[rep] = t_model.layer2[1].bn2.weight.data[m].mean(dim=0)
                t_model.layer2[1].bn2.bias.data[rep] = t_model.layer2[1].bn2.bias.data[m].mean(dim=0)
                norm = np.sqrt(len(m))
                for x in m:
                    if x != rep:
                        t_model.layer3[0].conv1.weight.data[:, rep] += t_model.layer3[0].conv1.weight.data[:, x] / norm
                        if len(t_model.layer3[0].shortcut) > 0:
                            t_model.layer3[0].shortcut[0].weight.data[:, rep] += t_model.layer3[0].shortcut[0].weight.data[:, x] / norm

        # Stage 3 Clustering
        if len(rem3) > 0 and k_c3 > 0:
            c3 = AgglomerativeClustering(n_clusters=min(k_c3, len(rem3)), metric='cosine', linkage='average')
            labels3 = c3.fit_predict(fp_s3_safe[rem3])
            for cid in range(k_c3):
                m = rem3[labels3 == cid]
                if len(m) == 0: continue
                rep = find_channel_medoid(fp_s3_safe, mean_act_s3, m)
                kept3.add(rep)
                t_model.layer3[1].conv2.weight.data[rep] = t_model.layer3[1].conv2.weight.data[m].mean(dim=0)
                t_model.layer3[1].bn2.weight.data[rep] = t_model.layer3[1].bn2.weight.data[m].mean(dim=0)
                t_model.layer3[1].bn2.bias.data[rep] = t_model.layer3[1].bn2.bias.data[m].mean(dim=0)
                norm = np.sqrt(len(m))
                for x in m:
                    if x != rep:
                        t_model.layer4[0].conv1.weight.data[:, rep] += t_model.layer4[0].conv1.weight.data[:, x] / norm
                        if len(t_model.layer4[0].shortcut) > 0:
                            t_model.layer4[0].shortcut[0].weight.data[:, rep] += t_model.layer4[0].shortcut[0].weight.data[:, x] / norm

        # Stage 4 Clustering
        if len(rem4) > 0 and k_c4 > 0:
            c4 = AgglomerativeClustering(n_clusters=min(k_c4, len(rem4)), metric='cosine', linkage='average')
            labels4 = c4.fit_predict(fp_s4_safe[rem4])
            for cid in range(k_c4):
                m = rem4[labels4 == cid]
                if len(m) == 0: continue
                rep = find_channel_medoid(fp_s4_safe, mean_act_s4, m)
                kept4.add(rep)
                t_model.layer4[1].conv2.weight.data[rep] = t_model.layer4[1].conv2.weight.data[m].mean(dim=0)
                t_model.layer4[1].bn2.weight.data[rep] = t_model.layer4[1].bn2.weight.data[m].mean(dim=0)
                t_model.layer4[1].bn2.bias.data[rep] = t_model.layer4[1].bn2.bias.data[m].mean(dim=0)
                norm = np.sqrt(len(m))
                for x in m:
                    if x != rep:
                        t_model.fc.weight.data[:, rep] += t_model.fc.weight.data[:, x] / norm

        sub_net = create_sub_resnet(t_model, kept1, kept2, kept3, kept4)
        
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

        total_kept_count = len(kept1) + len(kept2) + len(kept3) + len(kept4)
        return zs_acc, zs_agr, ft_acc, ft_agr, total_kept_count, t_zs, t_ft

    res = {'our_zs_acc': [], 'our_zs_agr': [], 'our_ft_acc': [], 'our_ft_agr': [],
           'da_zs_acc': [], 'da_zs_agr': [], 'da_ft_acc': [], 'da_ft_agr': [],
           'pct_kept': [], 'our_zs_t': [], 'our_ft_t': [], 'da_zs_t': [], 'da_ft_t': []}

    for b in target_budgets:
        t_b_start = time.time()
        o_za, o_zg, o_fa, o_fg, total_k, o_tz, o_tf = build_conv_core(b, use_causal=True)
        d_za, d_zg, d_fa, d_fg, _, d_tz, d_tf = build_conv_core(total_k, use_causal=False)
        
        pct = (total_k / TOTAL_CHANNELS) * 100.0
        res['pct_kept'].append(pct)
        res['our_zs_acc'].append(o_za); res['our_zs_agr'].append(o_zg)
        res['our_ft_acc'].append(o_fa); res['our_ft_agr'].append(o_fg)
        res['da_zs_acc'].append(d_za); res['da_zs_agr'].append(d_zg)
        res['da_ft_acc'].append(d_fa); res['da_ft_agr'].append(d_fg)
        res['our_zs_t'].append(o_tz); res['our_ft_t'].append(o_tf)
        res['da_zs_t'].append(d_tz); res['da_ft_t'].append(d_tf)
        
        b_time = time.time() - t_b_start
        print(f"{get_elapsed_str()}     -> Budget {pct:4.1f}% ({total_k:3d} ch) | Our FT: {o_fa:.2f}% | DA FT: {d_fa:.2f}% ({b_time:.1f}s)")

    return res

# =====================================================================
# 4. Main Benchmark Orchestrator
# =====================================================================
print(f"=== STARTING CIFAR-10 RESNET-18 BENCHMARK ({len(SEEDS)} SEEDS) ===")
all_cifar_runs = []
for idx, s in enumerate(SEEDS):
    all_cifar_runs.append(run_cifar_seed_benchmark(idx, s))

pct_axis = np.mean([r['pct_kept'] for r in all_cifar_runs], axis=0)

def aggregate_cifar(key):
    data = np.array([r[key] for r in all_cifar_runs])
    return np.mean(data, axis=0), np.std(data, axis=0)

our_ft_acc_m, our_ft_acc_s = aggregate_cifar('our_ft_acc')
our_ft_agr_m, our_ft_agr_s = aggregate_cifar('our_ft_agr')
our_zs_acc_m, our_zs_acc_s = aggregate_cifar('our_zs_acc')
our_zs_agr_m, our_zs_agr_s = aggregate_cifar('our_zs_agr')

da_ft_acc_m, da_ft_acc_s = aggregate_cifar('da_ft_acc')
da_ft_agr_m, da_ft_agr_s = aggregate_cifar('da_ft_agr')
da_zs_acc_m, da_zs_acc_s = aggregate_cifar('da_zs_acc')
da_zs_agr_m, da_zs_agr_s = aggregate_cifar('da_zs_agr')

our_zs_t_m, _ = aggregate_cifar('our_zs_t')
our_ft_t_m, _ = aggregate_cifar('our_ft_t')
da_zs_t_m, _ = aggregate_cifar('da_zs_t')
da_ft_t_m, _ = aggregate_cifar('da_ft_t')

# =====================================================================
# 5. Plotting
# =====================================================================
print(f"\n{get_elapsed_str()} Generating Benchmark Figures...")
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6))

def plot_cifar_band(ax, x, mean, std, fmt, color, label):
    ax.plot(x, mean, fmt, color=color, linewidth=2.0, label=label)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

plot_cifar_band(ax1, pct_axis, our_ft_agr_m, our_ft_agr_s, 'D-', '#008080', 'Our Method (KD FT + EMA)')
plot_cifar_band(ax1, pct_axis, our_zs_agr_m, our_zs_agr_s, 's-.', '#1f77b4', 'Our Method (Zero-Shot)')
plot_cifar_band(ax1, pct_axis, da_ft_agr_m, da_ft_agr_s, '^-', '#6a3d9a', 'DeepAbstract (KD FT)')
plot_cifar_band(ax1, pct_axis, da_zs_agr_m, da_zs_agr_s, 'v-.', '#9467bd', 'DeepAbstract (Zero-Shot)')
ax1.set_xlabel('Percentage of Stage Channels Kept (%)')
ax1.set_ylabel('Agreement with ResNet-18 Teacher (%)')
ax1.set_title('CIFAR-10 Agreement vs. % Channels Kept')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='lower right', fontsize=8)

plot_cifar_band(ax2, pct_axis, our_ft_acc_m, our_ft_acc_s, 'D-', '#008080', 'Our Method (KD FT + EMA)')
plot_cifar_band(ax2, pct_axis, our_zs_acc_m, our_zs_acc_s, 's-.', '#2ca02c', 'Our Method (Zero-Shot)')
plot_cifar_band(ax2, pct_axis, da_ft_acc_m, da_ft_acc_s, '^-', '#6a3d9a', 'DeepAbstract (KD FT)')
plot_cifar_band(ax2, pct_axis, da_zs_acc_m, da_zs_acc_s, 'v-.', '#9467bd', 'DeepAbstract (Zero-Shot)')
ax2.set_xlabel('Percentage of Stage Channels Kept (%)')
ax2.set_ylabel('Test Accuracy (%)')
ax2.set_title('CIFAR-10 Test Accuracy vs. % Channels Kept')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(loc='lower right', fontsize=8)

# Grouped Bar Chart: Zero-Shot vs Fine-Tuned
eval_groups = ['Zero-Shot', 'Fine-Tuned']
bw = 0.35
x_idx = np.arange(len(eval_groups))

our_times = [np.mean(our_zs_t_m), np.mean(our_ft_t_m)]
da_times = [np.mean(da_zs_t_m), np.mean(da_ft_t_m)]

ax3.bar(x_idx - bw/2, our_times, bw, label='Our Method', color='#008080', edgecolor='black')
ax3.bar(x_idx + bw/2, da_times, bw, label='DeepAbstract', color='#6a3d9a', edgecolor='black')

ax3.set_ylabel('Execution Time per Run (ms)')
ax3.set_title('ResNet-18 Average Execution Time')
ax3.set_xticks(x_idx)
ax3.set_xticklabels(eval_groups)
ax3.grid(True, axis='y', linestyle='--', alpha=0.5)
ax3.legend(loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig('cifar10_resnet18_benchmark.png', dpi=300)
plt.close()

print(f"{get_elapsed_str()} Benchmark complete. Output plot saved to 'cifar10_resnet18_benchmark.png'")