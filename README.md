# Retinal Artery–Vein Segmentation

This project implements **multi-class retinal artery–vein (A/V) segmentation** using deep learning models built with **PyTorch** and **segmentation_models_pytorch**.

The goal is to classify retinal vessels in fundus images into:

* **Background**
* **Artery**
* **Vein**
* (Optional) Crossing / uncertain class

The project compares multiple encoder–decoder architectures and initialization strategies.

---

# 📁 Project Structure

```
.
├── unet_with_efficientnet_b4.ipynb
├── unet_with_resnet34.ipynb
├── unetpp.ipynb
└── README.md
```

---

# 📓 Notebooks Overview

## 1️⃣ unet_with_efficientnet_b4.ipynb

**Model:** U-Net
**Encoder:** EfficientNet-B4
**Framework:** segmentation_models_pytorch

### Features:

* Pretrained encoder (ImageNet)
* Strong data augmentation
* Combined CrossEntropy + Dice loss
* Multi-dataset training support
* IoU and Dice evaluation
* Visualization of predictions

This model focuses on leveraging a high-capacity encoder for improved vessel feature extraction.

---

## 2️⃣ unet_with_resnet34.ipynb

**Model:** U-Net
**Encoder:** ResNet34
**Framework:** segmentation_models_pytorch

### Features:

* Lightweight and efficient architecture
* Faster training compared to EfficientNet-B4
* Suitable for limited GPU memory
* Dice + CrossEntropy loss
* Multi-class segmentation

This notebook provides a strong baseline with lower computational cost.

---

## 3️⃣ unetpp.ipynb

**Model:** U-Net++
**Encoder:** EfficientNet-B4 (or configurable)

### Features:

* Nested skip connections (dense decoder)
* Improved feature fusion
* Better small vessel recovery
* Multi-stage training experiments
* Cross-dataset evaluation
* Visualization and metric logging

U-Net++ improves boundary refinement and thin vessel segmentation.

---

# 🧠 Methodology

## Data Processing

* Color-coded label conversion → class indices
* Green channel CLAHE enhancement
* Synchronized image–mask augmentation
* Optional FOV masking

## Loss Function

Combined loss:

```
Loss = 0.6 × CrossEntropy + 0.4 × Dice
```

* CrossEntropy → pixel-level accuracy
* Dice → overlap quality
* Class weighting for imbalance handling

## Evaluation Metrics

* Per-class Dice score
* Per-class IoU
* Mean IoU (excluding background)
* Cross-dataset testing

---

# 🧪 Datasets

The notebooks support training and evaluation on common retinal A/V datasets such as:

* DRIVE
* HRF
* LES-AV

Datasets must be organized into:

```
dataset/
├── images/
├── labels/
└── masks/   (optional FOV masks)
```

---

# ⚙️ Requirements

Install dependencies:

```bash
pip install torch torchvision
pip install segmentation-models-pytorch
pip install albumentations
pip install opencv-python
pip install matplotlib
pip install numpy
pip install tqdm
```

Recommended:

* Python 3.8+
* CUDA-enabled GPU

---

# 🚀 How to Run

1. Open a notebook:

   ```
   jupyter notebook
   ```
2. Configure dataset paths.
3. Select model architecture and hyperparameters.
4. Run all cells.
5. Check:

   * Validation metrics
   * Saved model checkpoints
   * Prediction visualizations

---

# 📊 Experiments

The project compares:

| Model   | Encoder         | Strength                   |
| ------- | --------------- | -------------------------- |
| U-Net   | ResNet34        | Lightweight baseline       |
| U-Net   | EfficientNet-B4 | Strong feature extractor   |
| U-Net++ | EfficientNet-B4 | Improved boundary modeling |

Evaluation focuses on:

* Generalization across datasets
* Thin vessel recovery
* Artery vs vein classification accuracy

---

# Research Motivation

Retinal artery–vein segmentation is important for:

* Cardiovascular risk assessment
* Hypertension analysis
* Diabetic retinopathy studies
* Automated ophthalmology screening systems

This project explores how architecture and initialization influence multi-class vessel classification performance.

---

# 🏆 Key Features

✔ Multi-class A/V segmentation

✔ Transfer learning from ImageNet

✔ Dice + CrossEntropy hybrid loss

✔ Strong augmentation pipeline

✔ Cross-dataset evaluation

✔ Visualization tools

✔ Modular architecture design

If you want, I can also generate:

* A more academic-style README
* A GitHub-ready version with badges
* A simplified version for coursework submission
* Or a version tailored for a research paper submission 🚀
