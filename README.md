MAY 2026


# Deepfake Detection with ResNet-18 & PyTorch

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Project Overview
This project is a high-performance image classification system designed to detect AI-generated (Deepfake) faces. Using **Transfer Learning** and the **ResNet-18** architecture, the model was trained on a massive dataset of 140,000 images to identify digital artifacts and inconsistencies typical of GAN-generated imagery.

The system is optimized for **Apple Silicon (M-series)** hardware, leveraging the Metal Performance Shaders (MPS) backend for GPU-accelerated training and inference.

---

## Performance Metrics
* **Final Test Accuracy:** `85.08%`
* **Evaluation Set:** 20,000 previously unseen images.
* **Hardware:** Trained on MacBook Pro (MPS/GPU).
* **Training Time:** 5 Epochs with a steady loss convergence from `0.47` to `0.38`.

---

## Technical Architecture

### 1. Data Pipeline
* **Dataset:** Real vs. Fake Faces (140k images).
* **Preprocessing:** * Dynamic resizing to `224x224` pixels.
    * Normalization based on ImageNet statistics ($\mu, \sigma$).
    * Efficient data loading via PyTorch `DataLoaders`.

### 2. Model Design
* **Base:** Pre-trained ResNet-18 (Weights: `DEFAULT`).
* **Custom Classification Head:**
    * `Linear(512 -> 256)`
    * `ReLU` Activation
    * `Dropout(0.4)` for regularization and overfit prevention.
    * `Linear(256 -> 2)`
    * `LogSoftmax` for stable probability output.

### 3. Optimization Strategy
* **Optimizer:** Adam ($LR = 0.001$).
* **Loss Function:** `NLLLoss` (Negative Log Likelihood).
* **Device:** `torch.device("mps")` for Apple GPU acceleration.

---

##  Project Structure
```text
deepfake-detector/
├── src/
│   ├── data_loader.py   # Dataset loading & transformations
│   ├── model.py         # Neural Network architecture
│   ├── train.py         # Training loop & logic
│   └── evaluate.py      # Performance testing script
├── models/
│   └── deepfake_detector.pth # Trained model weights
├── FINAL_PROJECT_LOG.md  # Detailed development journal
└── README.md             # Project documentation