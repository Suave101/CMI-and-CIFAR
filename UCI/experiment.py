import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# -------------------------------------------------------------------
# 1. Architecture Modified for CIFAR-10 (Option 1)
# -------------------------------------------------------------------
def get_cifar_resnet18(num_classes=10):
    model = torchvision.models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, num_classes)
    return model

# -------------------------------------------------------------------
# 2. Training Loop with Test Set Accuracy Evaluation
# -------------------------------------------------------------------
def train_and_evaluate(model, epochs=10, batch_size=128, lr=0.1, device="cuda"):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.to(device)
    print(f"--- Training ResNet-18 on CIFAR-10 ({epochs} Epochs) ---")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        # Evaluate accuracy on test set
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_accuracy = 100. * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {running_loss/len(trainloader):.4f} | Test Accuracy: {test_accuracy:.2f}%")

    print(f"\nFinal Test Set Accuracy: {test_accuracy:.2f}%")
    return model, testloader

# -------------------------------------------------------------------
# 3. Neuron Importance Calculation
# Importance = (Output with Neuron Present) - (Output without Neuron)
# -------------------------------------------------------------------
def compute_input_neuron_importance(model, image_tensor, target_class=None, batch_size=256):
    model.eval()
    device = image_tensor.device
    _, C, H, W = image_tensor.shape
    num_neurons = C * H * W

    # Step A: Output with neuron present (Baseline forward pass)
    with torch.no_grad():
        clean_logits = model(image_tensor)
        if target_class is None:
            target_class = clean_logits.argmax(dim=-1).item()
        output_with_neuron_present = clean_logits[0, target_class].item()

    print(f"\nTarget Class: {target_class} | Output with neuron present: {output_with_neuron_present:.4f}")

    # Step B: Zero out each input neuron one by one
    ablated_batch = image_tensor.repeat(num_neurons, 1, 1, 1)
    flat_ablated = ablated_batch.view(num_neurons, -1)
    flat_ablated[torch.arange(num_neurons), torch.arange(num_neurons)] = 0.0
    ablated_batch = flat_ablated.view(num_neurons, C, H, W)

    output_without_neuron = []
    progress_bar = tqdm(
        range(0, num_neurons, batch_size),
        desc="Ablating Input Neurons",
        unit="batch",
        total=(num_neurons + batch_size - 1) // batch_size
    )

    # Step C: Output without neuron
    with torch.no_grad():
        for i in progress_bar:
            chunk = ablated_batch[i : i + batch_size].to(device)
            logits = model(chunk)
            output_without_neuron.append(logits[:, target_class])

    output_without_neuron = torch.cat(output_without_neuron, dim=0)

    # Step D: Importance = (Output with neuron present) - (Output without neuron)
    importance_map = (output_with_neuron_present - output_without_neuron).view(C, H, W)
    return importance_map, target_class

# -------------------------------------------------------------------
# 4. Save Visualizations to PNG
# -------------------------------------------------------------------
def save_importance_heatmap(importance_map, output_filename="trained_cifar10_importance.png"):
    imp_cpu = importance_map.cpu().numpy()
    spatial_heatmap = imp_cpu.sum(axis=0)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    im0 = axes[0].imshow(spatial_heatmap, cmap="coolwarm")
    axes[0].set_title("Aggregated Spatial")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    channel_names = ["Red Channel", "Green Channel", "Blue Channel"]
    for idx, name in enumerate(channel_names):
        im = axes[idx + 1].imshow(imp_cpu[idx], cmap="coolwarm")
        axes[idx + 1].set_title(name)
        axes[idx + 1].axis("off")
        fig.colorbar(im, ax=axes[idx + 1], fraction=0.046, pad=0.04)

    plt.suptitle("Causal Input Neuron Importance (Trained ResNet-18)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved successfully to '{output_filename}'")

# -------------------------------------------------------------------
# 5. Main Execution Flow
# -------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Instantiate and train model (Option 1)
    model = get_cifar_resnet18()
    model, testloader = train_and_evaluate(model, epochs=10, device=device)

    # 2. Extract a real image from the CIFAR-10 test set
    test_iter = iter(testloader)
    images, labels = next(test_iter)
    sample_image = images[0:1].to(device)  # Shape: (1, 3, 32, 32)

    # 3. Compute neuron importance map
    importance_map, target_cls = compute_input_neuron_importance(
        model, 
        sample_image, 
        batch_size=256
    )

    # 4. Save PNG output
    save_importance_heatmap(importance_map, output_filename="trained_cifar10_importance.png")
