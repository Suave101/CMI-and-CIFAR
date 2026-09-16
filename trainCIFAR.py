import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedShuffleSplit

SEEDS = [42, 101, 2024, 7, 99]
CHECKPOINT_DIR = "./checkpoints"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_teachers_infinitely():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    full_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    targets = np.array(full_train.targets)

    for seed in SEEDS:
        print(f"\n=== Training Teacher Model Infinitely (Seed {seed}) ===")
        print("Press Ctrl+C at any time to stop training this seed.\n")
        set_seed(seed)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, _ = next(sss.split(np.zeros(len(targets)), targets))
        train_loader = DataLoader(Subset(full_train, train_idx), batch_size=128, shuffle=True)

        teacher_model = CIFARConvNet().to(device)
        optimizer = optim.Adam(teacher_model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        latest_ckpt_path = os.path.join(CHECKPOINT_DIR, f"teacher_seed_{seed}_latest.pth")
        start_epoch = 1

        # Resume if a saved checkpoint exists
        if os.path.exists(latest_ckpt_path):
            ckpt = torch.load(latest_ckpt_path, map_location=device)
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                teacher_model.load_state_dict(ckpt['state_dict'])
                if 'optimizer' in ckpt:
                    optimizer.load_state_dict(ckpt['optimizer'])
                start_epoch = ckpt.get('epoch', 0) + 1
                print(f"Resuming seed {seed} starting at Epoch {start_epoch}...")
            else:
                teacher_model.load_state_dict(ckpt)
                print(f"Loaded raw state dict for seed {seed}.")

        teacher_model.train()
        epoch = start_epoch
        try:
            while True:
                running_loss = 0.0
                for x_b, y_b in train_loader:
                    x_b, y_b = x_b.to(device), y_b.to(device)
                    optimizer.zero_grad()
                    out = teacher_model(x_b)
                    loss = criterion(out, y_b)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                avg_loss = running_loss / len(train_loader)
                print(f"  [Seed {seed}] Epoch {epoch:04d} | Loss: {avg_loss:.4f}")

                checkpoint_data = {
                    'epoch': epoch,
                    'state_dict': teacher_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': avg_loss
                }

                # Save per-epoch snapshot
                epoch_ckpt_path = os.path.join(CHECKPOINT_DIR, f"teacher_seed_{seed}_epoch_{epoch}.pth")
                torch.save(checkpoint_data, epoch_ckpt_path)

                # Overwrite/update latest snapshot
                torch.save(checkpoint_data, latest_ckpt_path)

                epoch += 1

        except KeyboardInterrupt:
            print(f"\n[Interrupted] Stopped training seed {seed} at Epoch {epoch - 1}. Last checkpoint saved.")
            cont = input("Move to next seed? [y/N]: ").strip().lower()
            if cont != 'y':
                print("Exiting training process.")
                sys.exit(0)

if __name__ == "__main__":
    train_teachers_infinitely()