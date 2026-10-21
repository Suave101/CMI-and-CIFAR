import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# 1. Dataset Setup (UCI Digits: 1437 train, 360 test)
# =====================================================================
digits = load_digits()
X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=360, random_state=42, stratify=y
)

mean, std = (
    X_train.mean(axis=0, keepdims=True),
    X_train.std(axis=0, keepdims=True) + 1e-8,
)
X_train_std = (X_train - mean) / std
X_test_std = (X_test - mean) / std

X_train_t = torch.tensor(X_train_std, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test_std, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# Save processed data tensors for downstream scripts
torch.save(
    {
        "X_train": X_train_t,
        "y_train": y_train_t,
        "X_test": X_test_t,
        "y_test": y_test_t,
    },
    "dataset.pt",
)


# =====================================================================
# 2. Model Definition & Training (256x128 = 384 hidden neurons)
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
        out = self.fc3(h2)
        return out


model = MLP()
optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(100):
    optimizer.zero_grad()
    loss = criterion(model(X_train_t), y_train_t)
    loss.backward()
    optimizer.step()

model.eval()

# Save trained weights
torch.save(model.state_dict(), "mlp_model.pth")


def get_accuracy(model_obj, X_t, y_t):
    with torch.no_grad():
        preds = model_obj(X_t).argmax(dim=1)
        acc = (preds == y_t).float().mean().item()
    return acc


print("=== BASELINE MODEL TRAINED & SAVED ===")
print(f"Training Accuracy: {get_accuracy(model, X_train_t, y_train_t) * 100:.2f}%")
print(f"Test Accuracy:     {get_accuracy(model, X_test_t, y_test_t) * 100:.2f}%")
print("Saved artifacts: 'mlp_model.pth' and 'dataset.pt'")
