#Evaluation Script - Baseline vs Improved ResNet-18

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# Recorded Results (from training runs) 
epochs_baseline = list(range(1, 11))
epochs_improved = list(range(1, 21))

baseline_losses = [
    0.9810, 0.6299, 0.4812, 0.4112, 0.3261,
    0.2519, 0.2366, 0.1731, 0.1400, 0.1286
]
baseline_accuracies = [
    77.17, 78.71, 78.81, 79.40, 80.43,
    81.52, 79.72, 80.13, 80.71, 81.00
]

improved_losses = [
    1.2247, 0.8770, 0.7685, 0.6800, 0.6498,
    0.6031, 0.5380, 0.5110, 0.4710, 0.4419,
    0.4108, 0.3821, 0.3535, 0.3286, 0.3129,
    0.3012, 0.2822, 0.2668, 0.2581, 0.2538
]
improved_accuracies = [
    71.02, 73.47, 77.61, 80.41, 76.93,
    80.41, 81.61, 82.94, 82.61, 82.95,
    84.48, 84.10, 85.46, 85.45, 85.38,
    86.11, 85.70, 86.41, 86.38, 86.33
]

baseline_f1 = [0.84, 0.89, 0.76, 0.65, 0.80, 0.72, 0.86, 0.86, 0.88, 0.85]
improved_f1 = [0.89, 0.93, 0.85, 0.73, 0.87, 0.75, 0.90, 0.90, 0.90, 0.91]


def plot_loss_comparison():
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_baseline, baseline_losses, marker='o',
             color='steelblue', linewidth=2, label='Baseline (10 epochs)')
    plt.plot(epochs_improved, improved_losses, marker='s',
             color='darkorange', linewidth=2, label='Improved (20 epochs)')
    plt.title("Training Loss: Baseline vs Improved", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("../results/chart1_loss_comparison.png", dpi=150)
    plt.close()
    print("Saved chart1_loss_comparison.png")


def plot_accuracy_comparison():
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_baseline, baseline_accuracies, marker='o',
             color='steelblue', linewidth=2, label='Baseline (10 epochs)')
    plt.plot(epochs_improved, improved_accuracies, marker='s',
             color='darkorange', linewidth=2, label='Improved (20 epochs)')
    plt.axhline(y=81.00, color='steelblue', linestyle='--', alpha=0.5,
                label='Baseline Final (81.00%)')
    plt.axhline(y=86.33, color='darkorange', linestyle='--', alpha=0.5,
                label='Improved Final (86.33%)')
    plt.title("Validation Accuracy: Baseline vs Improved", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("../results/chart2_accuracy_comparison.png", dpi=150)
    plt.close()
    print("Saved chart2_accuracy_comparison.png")


def plot_perclass_f1():
    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, baseline_f1, width, label='Baseline',
                   color='steelblue', alpha=0.85)
    bars2 = ax.bar(x + width/2, improved_f1, width, label='Improved',
                   color='darkorange', alpha=0.85)

    ax.set_title("Per-Class F1 Score: Baseline vs Improved", fontsize=14, fontweight='bold')
    ax.set_xlabel("Class")
    ax.set_ylabel("F1 Score")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha='right')
    ax.set_ylim(0.5, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    for bars in (bars1, bars2):
        for bar in bars:
            ax.annotate(f'{bar.get_height():.2f}',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig("../results/chart3_perclass_f1.png", dpi=150)
    plt.close()
    print("Saved chart3_perclass_f1.png")


def plot_confusion_matrix(model_path="../results/improved_model.pth"):
    #Loads the saved improved model and generates a confusion matrix.
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=False, num_workers=2)

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_percent, annot=True, fmt='.1f',
                xticklabels=classes, yticklabels=classes,
                cmap='Blues', linewidths=0.5)
    plt.title("Confusion Matrix — Improved Model (%)", fontsize=14, fontweight='bold')
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig("../results/chart4_confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved chart4_confusion_matrix.png")


def plot_overall_metrics():
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    baseline_scores = [0.81, 0.81, 0.81, 0.81]
    improved_scores = [0.8633, 0.86, 0.86, 0.86]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, baseline_scores, width, label='Baseline',
                   color='steelblue', alpha=0.85)
    bars2 = ax.bar(x + width/2, improved_scores, width, label='Improved',
                   color='darkorange', alpha=0.85)

    ax.set_title("Overall Performance Metrics: Baseline vs Improved", fontsize=14, fontweight='bold')
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0.70, 0.95)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    for bars in (bars1, bars2):
        for bar in bars:
            ax.annotate(f'{bar.get_height():.2f}',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig("../results/chart5_overall_metrics.png", dpi=150)
    plt.close()
    print("Saved chart5_overall_metrics.png")


if __name__ == "__main__":
    plot_loss_comparison()
    plot_accuracy_comparison()
    plot_perclass_f1()
    plot_overall_metrics()
    plot_confusion_matrix()  # requires improved_model.pth in ../results/
    print("\nAll evaluation charts generated successfully!")