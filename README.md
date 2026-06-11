# 🔬 DermaScan — Skin Lesion Classification & Explainability System

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/AWS_EC2-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white"/>
  <img src="https://img.shields.io/badge/ResNet--50-Transfer_Learning-blue?style=for-the-badge"/>
</p>

<p align="center">
  <a href="http://52.66.12.200:8501/" target="_blank">
    <img src="https://img.shields.io/badge/🚀 Live Demo-Click Here-brightgreen?style=for-the-badge"/>
  </a>
</p>

> A deep learning system for multi-class skin cancer lesion classification using ResNet-50 transfer learning, with Grad-CAM visual explanations deployed as a Streamlit web app on AWS EC2.

---

## 📌 Overview

DermaScan classifies dermoscopy images into **7 skin lesion categories** from the [HAM10000 dataset](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection), achieving **80% test accuracy** with **81% weighted F1-score**.

The system also generates **Grad-CAM heatmaps** that highlight the exact regions the model focused on — making predictions interpretable and clinically meaningful.

---

## 🏷️ Lesion Categories

| Abbreviation | Full Name |
|---|---|
| `akiec` | Actinic Keratoses / Intraepithelial Carcinoma |
| `bcc` | Basal Cell Carcinoma |
| `bkl` | Benign Keratosis-like Lesions |
| `df` | Dermatofibroma |
| `mel` | Melanoma |
| `nv` | Melanocytic Nevi |
| `vasc` | Vascular Lesions |

---

## 📊 Model Performance

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| akiec | 0.42 | 0.85 | 0.57 | 33 |
| bcc | 0.59 | 0.82 | 0.69 | 51 |
| bkl | 0.71 | 0.59 | 0.65 | 110 |
| df | 0.58 | 0.92 | 0.71 | 12 |
| mel | 0.53 | 0.62 | 0.57 | 111 |
| **nv** | **0.95** | **0.86** | **0.90** | **671** |
| vasc | 0.71 | 0.86 | 0.77 | 14 |
| | | | | |
| **Accuracy** | | | **0.80** | **1002** |
| Macro Avg | 0.64 | 0.79 | 0.69 | 1002 |
| **Weighted Avg** | **0.83** | **0.80** | **0.81** | **1002** |

### Confusion Matrix

<img width="871" height="673" alt="image" src="https://github.com/user-attachments/assets/1f52bbbf-98af-48fc-abfe-090fa39a1d0e" />


> The model performs strongly on `nv` (dominant class, F1: 0.90) and `df` (rare class, recall: 0.92). The main confusion occurs between visually similar classes: `mel` ↔ `nv` and `bkl` ↔ `akiec` — a known challenge in dermoscopy classification.

---

## 🏗️ Architecture

```
Input Image (224×224×3)
        │
        ▼
┌─────────────────────┐
│   ResNet-50 Backbone │  ← Pretrained on ImageNet (frozen early layers)
│   (Feature Extractor)│
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Custom Classification│
│       Head           │
│  FC → ReLU → Dropout │
│    → FC (7 classes)  │
└─────────────────────┘
        │
        ▼
  Softmax Output (7 classes)
        │
        ▼
┌─────────────────────┐
│     Grad-CAM         │  ← Saliency heatmap over input image
│  (last conv layer)   │
└─────────────────────┘
```

---

## ⚙️ Technical Details

### Transfer Learning Strategy
- **Backbone:** ResNet-50 pretrained on ImageNet
- **Fine-tuning:** Early layers frozen; later residual blocks unfrozen for domain adaptation
- **Classification Head:** `AdaptiveAvgPool → FC(2048→512) → ReLU → Dropout(0.5) → FC(512→7)`
- **Loss Function:** CrossEntropyLoss with class weights to handle severe class imbalance (`nv` dominates at ~67% of samples)
- **Optimizer:** Adam with learning rate scheduling

### Grad-CAM Explainability
- Hooks into the **last convolutional layer** of ResNet-50
- Computes **gradient-weighted class activation maps**
- Overlays heatmap on original image to show discriminative regions
- Helps clinicians verify whether the model attends to the actual lesion

### Class Imbalance Handling
- HAM10000 is heavily skewed — `nv` has 6705 samples vs `df` with only 115
- Addressed via **weighted loss**, **oversampling**, and **augmentation**

---

## 🚀 Getting Started

### Prerequisites
```bash
python >= 3.8
torch >= 1.12
torchvision
streamlit
opencv-python
numpy
Pillow
matplotlib
```

### Installation

```bash
git clone https://github.com/Dev-Maheshwari7/skin-cancer-transfer-learning.git
cd skin-cancer-transfer-learning
pip install -r requirements.txt
```

### Run Locally

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

---

## 🖥️ Live Demo

**Try it here → [http://52.66.12.200:8501/](http://52.66.12.200:8501/)**

Upload any dermoscopy image and get:
- Predicted lesion class with confidence scores
- Grad-CAM heatmap showing where the model looked

> ⚠️ This is a research tool and **not a medical diagnostic device**. Always consult a qualified dermatologist.

---

## 📁 Project Structure

```
skin-cancer-transfer-learning/
│
├── SkinCancerPrediction.ipynb   # Model training, evaluation & Grad-CAM experiments
├── app.py                       # Streamlit web app
├── requirements.txt
└── resnet50_skin_first.pth      # Trained ResNet-50 weights
└── README.md
```

---

## 🔬 Dataset

**HAM10000** (*Human Against Machine with 10000 training images*)
- 10,015 dermoscopy images across 7 lesion classes
- Source: [Kaggle — Skin Lesion Analysis](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection)
- Split: 80% train / 10% val / 10% test

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Deep Learning Framework | PyTorch |
| Model Architecture | ResNet-50 (Transfer Learning) |
| Explainability | Grad-CAM |
| Web App | Streamlit |
| Deployment | AWS EC2 |

---

## 👤 Author

**Dev Maheshwari**
- GitHub: [@Dev-Maheshwari7](https://github.com/Dev-Maheshwari7)
- HuggingFace: [@Dev3771](https://huggingface.co/Dev3771)

---

## ⚠️ Disclaimer

This project is intended for **educational and research purposes only**. It is not a substitute for professional medical diagnosis. Predictions should not be used for clinical decision-making without validation by a licensed dermatologist.
