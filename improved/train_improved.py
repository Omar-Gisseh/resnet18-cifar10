# Improved ResNet-18 on CIFAR-10
# Enhancements: Data Augmentation + LR Scheduling

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np
import time

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# Improved Data Loading with Augmentation 
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),           # Randomly flip images
    transforms.RandomCrop(32, padding=4),        # Random crop with padding
    transforms.ColorJitter(                      # Random color changes
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(                        # Better normalization
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
])

# Test set gets NO augmentation — only normalization
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=train_transform)
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=test_transform)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=64, shuffle=False, num_workers=2)

classes = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']

# Model
model = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4       # L2 regularization
)

# Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=20                # Smoothly decays LR over 20 epochs
)

# Training
EPOCHS = 20                 # More epochs for better convergence
train_losses = []
val_accuracies = []
lr_history = []

print("\nStarting improved training...\n")
start_time = time.time()

for epoch in range(EPOCHS):
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

    # Step the scheduler
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    lr_history.append(current_lr)

    avg_loss = running_loss / len(trainloader)
    train_losses.append(avg_loss)

    # Validation accuracy
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    val_accuracies.append(acc)
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}% | LR: {current_lr:.6f}")

total_time = time.time() - start_time
print(f"\nTraining complete in {total_time/60:.2f} minutes")

# Final Evaluation 
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\n--- Improved Classification Report ---")
print(classification_report(all_labels, all_preds, target_names=classes))

# Plot Results
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.plot(train_losses, marker='o', color='steelblue')
plt.title("Improved Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1, 3, 2)
plt.plot(val_accuracies, marker='o', color='darkorange')
plt.axhline(y=81.00, color='red', linestyle='--', label='Baseline (81%)')
plt.title("Improved Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(lr_history, marker='o', color='green')
plt.title("Learning Rate Schedule")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")

plt.tight_layout()
plt.savefig("improved_results.png")
plt.show()
print("\nDone!")

# Save Model
torch.save(model.state_dict(), "improved_model.pth")
print("Improved model saved!")