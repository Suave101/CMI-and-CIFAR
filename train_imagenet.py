import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet18

# Configuration
EPOCHS = 90
BATCH_SIZE = 256  # Adjust based on GPU memory (e.g., 64/128 per GPU)
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
IMAGENET_DIR = "./imagenet"  # Must contain 'train' and 'val' subfolders
WEIGHT_DIR = "weights"
SAVE_PATH = os.path.join(WEIGHT_DIR, "resnet18_imagenet1k_best.pth")

os.makedirs(WEIGHT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. ImageNet Data Augmentation
traindir = os.path.join(IMAGENET_DIR, "train")
valdir = os.path.join(IMAGENET_DIR, "val")

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    ),
)

val_dataset = datasets.ImageFolder(
    valdir,
    transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    ),
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True
)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True
)

# 2. Standard ResNet-18 Initialization
model = resnet18(weights=None, num_classes=1000)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


# 3. Accuracy Evaluation Helper
def accuracy(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# 4. Training Loop
best_top1 = 0.0
start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    scheduler.step()

    # Validation Phase
    model.eval()
    val_top1, val_top5, total_samples = 0.0, 0.0, 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))

            batch_sz = inputs.size(0)
            val_top1 += prec1.item() * batch_sz
            val_top5 += prec5.item() * batch_sz
            total_samples += batch_sz

    epoch_top1 = val_top1 / total_samples
    epoch_top5 = val_top5 / total_samples

    # Checkpoint Logic
    state_to_save = (
        model.module.state_dict()
        if isinstance(model, nn.DataParallel)
        else model.state_dict()
    )
    if epoch_top1 > best_top1:
        best_top1 = epoch_top1
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": state_to_save,
                "top1": best_top1,
                "top5": epoch_top5,
            },
            SAVE_PATH,
        )
        saved_str = f" [Saved Best Top-1: {best_top1:.2f}%]"
    else:
        saved_str = ""

    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(
        f"[{elapsed}] Epoch {epoch:02d}/{EPOCHS:02d} | Val Top-1: {epoch_top1:.2f}% | Val Top-5: {epoch_top5:.2f}%{saved_str}"
    )

print(
    f"\nTraining Complete. Best ImageNet Top-1 Accuracy: {best_top1:.2f}%. Saved to {SAVE_PATH}"
)
