MAY 2026

Log #1:

# 🛡️ Deepfake Detection Project: Development Log

**Author:** Priyaj Shrestha  
**Date:** May 2026  
**Status:** 🟢 Training Phase (Epoch 1 Complete)

---

## 📋 Project Overview
A machine learning pipeline designed to identify AI-generated faces. This project utilizes deep residual learning (ResNet) to detect subtle artifacts in synthetic imagery that are often invisible to the human eye.

## 💻 Technical Stack
* **Language:** Python 3.9
* **Deep Learning:** PyTorch, Torchvision
* **Optimization:** Apple Silicon MPS (Metal Performance Shaders) for GPU acceleration
* **Architecture:** ResNet-18 (Transfer Learning)

---

## 🛠️ Implementation Details

### 1. Data Pipeline
* **Dataset:** Kaggle 140k Real/Fake Faces.
* **Pre-processing:** * Images resized to `224x224`.
    * Normalized using ImageNet stats: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.
* **Loaders:** Implemented `DataLoader` with batch size of 32 and multi-set partitioning (Train/Val/Test).

### 2. Model Architecture
I utilized a **Transfer Learning** strategy to leverage weights from a model pre-trained on the ImageNet dataset.
* **Base:** ResNet-18 (Feature extractor frozen).
* **Modified Head:**
    * `Linear(512, 256)` -> `ReLU` -> `Dropout(p=0.4)` -> `Linear(256, 2)`.
    * `LogSoftmax` output for binary classification.

### 3. Training Configuration
* **Loss Function:** `NLLLoss` (Negative Log Likelihood).
* **Optimizer:** `Adam` (Learning Rate: 0.001).
* **Hardware:** Initialized `torch.device("mps")` to utilize MacBook Pro GPU cores.

---

## 📈 Current Metrics
| Metric | Value | Status |
| :--- | :--- | :--- |
| **Epoch 1 Avg Loss** | 0.4722 | ✅ Success |
| **Random Guess Baseline** | ~0.6931 | 📉 Beaten |
| **Hardware Device** | Apple MPS | 🚀 Active |

---

## 📂 Directory Structure
```text
deepfake-detector/
├── data/               # 140k JPG images (local only)
├── models/             # Exported .pth weights
└── src/
    ├── data_loader.py  # Image transformation logic
    ├── model.py        # Neural Network architecture
    └── train.py        # Training & Optimization loop