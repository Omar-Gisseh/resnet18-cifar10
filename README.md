# Fine-Tuning ResNet-18 on CIFAR-10

AI Final Group Project Option A: Enhancing an Existing AI Model

---

## Project Overview

This project investigates how targeted fine-tuning techniques can improve the performance of a pre-trained ResNet-18 model on the CIFAR-10 image classification benchmark. We establish a baseline using the out-of-the-box pre-trained model and then apply three enhancements; data augmentation, CIFAR-10-specific normalization, and cosine annealing learning rate scheduling to measure the improvement.

---

## Results Summary

| Model | Accuracy | Macro F1 | Epochs |
|-------|----------|----------|--------|
| Baseline | 81.00% | 0.81 | 10 |
| Improved | 86.33% | 0.86 | 20 |

**Improvement: +5.33 percentage points in overall accuracy**

---

## Requirements

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## How to Run

### Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com)
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Upload the script you want to run or copy its contents into a cell
4. Run all cells

### Local (CPU only)

```bash
python baseline/train_baseline.py
python improved/train_improved.py
python evaluation/evaluate.py
```

> **Note:** The CIFAR-10 dataset will be downloaded automatically on first run (~170MB).

---

## Improvements Applied

| Enhancement | Description |
|---|---|
| Data Augmentation | Random horizontal flip, random crop with padding, color jitter |
| Normalization | CIFAR-10-specific mean and std values |
| LR Scheduling | Cosine annealing over 20 epochs |
| Weight Decay | L2 regularization (1e-4) |

---

## References

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR.
- Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images.
- Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic Gradient Descent with Warm Restarts. ICLR.

---

## Authors

- Kingsley Ugochukwu Nnaemedo
- Omar Gisseh
