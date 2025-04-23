# Anomaly Detection Framework 🛠️

## Overview
A unified pipeline for self-supervised (SSL) pretraining, deep SVDD, and PaDiM anomaly detection on image data.
1. SSL pretraining gives a strong backbone.
2. SVDD uses that backbone for a fast, global anomaly detector (image-level).
3. PaDiM uses the same backbone but augments it with patch distributions for fine-grained, pixel-level heatmaps.
4. Including asic script using ImageNet‐pretrained ResNet to extract features and detect anomalies to compare speed and output to approach above

### Self-Supervised Learning (SSL)
 - Representation learning without labels. Real-world “normal” data often isn’t labeled, so we use a simple pretext task (predicting image rotations) to teach a backbone network useful visual features ​
 - Better features → better anomaly detection. A model that’s seen millions of rotations learns edges, textures, and shapes—so when we later train SVDD or PaDiM on top of those features, they’re more discriminative than starting from scratch or even ImageNet alone.

### Deep SVDD
 - One-class anomaly detection. SVDD learns a “hypersphere” in feature space that tightly encloses normal samples; anything that falls outside (farther than a learned center) is flagged anomalous
 - Simplicity & interpretability. You get a single distance score per image, easy thresholding, and clear notion of “distance to normal.” It’s lightweight and well-suited for small datasets of purely normal images.

### PaDiM (Patch-Distribution Modeling)
 - Pixel-level anomaly localization. SVDD gives a per-image score, but PaDiM builds a Gaussian+PCA model per spatial patch of a feature map, then measures patch-wise Mahalanobis distances ​padim. That yields a dense heatmap you can overlay on the input.
 - Richer modeling of local context. By modeling each patch’s distribution (and reducing dimensionality via PCA), PaDiM captures subtle structural deviations (e.g. cracks, texture changes) that a global SVDD score might miss.

---

## 🚀 Features

- **Basic ImageNet-based detection**: `simple_detection.py --image <IMG>` uses a frozen ResNet for quick, no-training anomaly scoring.  
- **Unified configuration** via `config.py` (no separate YAML/Python split).  
- **Single entrypoint training**: `train.py --mode {ssl,svdd,padim}`.  
- **Calibration** of anomaly‑score thresholds: `calibrate.py --mode {svdd,padim}`.  
- **Evaluation** on balanced/unbalanced splits: `evaluate.py --mode {svdd,padim}`.  
- **Unified inference**: `inference.py --mode {svdd,padim}` with single‑image `--image` or bulk‑sort `--sort_out`.  
- **Visualization**: heatmap overlays, score histograms, ROC/PR curves, and training‑loss plots.

---

## 📋 Requirements

- Python 3.8+  
- PyTorch 1.10+  
- torchvision, numpy, scikit-learn, matplotlib, opencv-python, tqdm, joblib, Pillow, PyYAML

Install via:

```bash
pip install -r requirements.txt
```

---

## 🗂️ Project Structure

```
├── config.py                # All basic configuration constants and paths
├── train.py                 # Unified training entrypoint for SSL, SVDD, or PaDiM
├── inference.py             # Fast single-image or bulk anomaly inference with SVDD or PaDiM
├── simple_detection.py      # Basic script using ImageNet‐pretrained ResNet to extract features and detect anomalies
├── trainers/
│   ├── ssl_trainer.py       # Self-supervised rotation prediction pretraining (SSL)
│   ├── svdd_trainer.py      # Deep SVDD one-class training on normal images
│   └── padim_trainer.py     # PaDiM training: per-patch Gaussian+PCA fitting
├── utils/
│   ├── datasets.py          # Dataset classes and DataLoader factories (CrackDataset, NormalDataset, etc.)
│   ├── visualize.py         # Plotting utilities (heatmaps, score histograms, ROC/PR curves, loss plots)
│   ├── feature_extractor.py # Backbone loading helpers (get_feature_backbone, get_classifier_backbone)
│   ├── calibrate.py         # Threshold calibration for SVDD and PaDiM (percentile-based)
│   └── evaluate.py          # Evaluation scripts: compute metrics and generate ROC/PR curves
├── checkpoints/
│   ├── ssl/                 # SSL model checkpoints and loss curves
│   ├── svdd/                # SVDD model checkpoints and loss curves
│   └── padim/               # PaDiM model files 
├── requirements.txt         # Python package dependencies
└── README.md                # Project overview, setup, and usage instructions
```

---

## ⚙️ Configuration (`config.py`)

Defines all paths and hyperparameters: Data paths, checkpoint dirs, model settings, training hyperparameters. Edit here; no other YAML needed.

---

## 📈 Training

### 1. SSL Pretraining

```bash
python train.py --mode ssl --epochs 10 --batch_size 64 --lr 1e-4
```

Output: `checkpoints/ssl/ssl_final.pth` & `ssl_loss_curve.png`.

### 2. Deep SVDD

```bash
python train.py --mode svdd --epochs 100 --batch_size 64 --lr 1e-5
```

Output: `checkpoints/svdd/svdd_final.pth` & `svdd_loss_curve.png`.

### 3. PaDiM

```bash
python train.py --mode padim --epochs 20 --batch_size 32
```

Output: `checkpoints/padim/padim_model.pt` & `padim_loss_curve.png`.

---

## 🎯 Calibration

```bash
python calibrate.py --mode svdd --percentile 95 --output thr_svdd.pkl
python calibrate.py --mode padim --percentile 95 --output thr_padim.pkl
```

---

## 🧪 Evaluation

```bash
python evaluate.py --mode svdd --threshold thr_svdd.pkl
python evaluate.py --mode padim --threshold thr_padim.pkl
```

Outputs: metrics & ROC/PR curves in `results/{svdd,padim}/…`.

---

## 🔍 Inference

### Basic detection
```bash
python simple_detection.py --image data/test.png
```

### Single-image

```bash
python inference.py --mode svdd --image data/test.png --threshold thr_svdd.pkl --output_overlay out/svdd_overlay.png
```

### Bulk sorting

```bash
python inference.py --mode padim --sort_out data/incoming_images/ --threshold thr_padim.pkl
```

---

## 📦 Notes

Use `--help` for any script. All paths come from `config.py`.

---
## Metrics

### SVDD
 - [Balanced] AUC=0.9751, Acc=0.9359, Prec=0.9486, Rec=0.9220, F1=0.9351
 - [Unbalanced] AUC=0.9896, Acc=0.9500, Prec=0.3750, Rec=1.0000, F1=0.5455
### PaDiM
 - [Balanced] AUC=0.9886, Acc=0.9454, Prec=0.9816, Rec=0.9080, F1=0.9434
 - [Unbalanced] AUC=0.9801, Acc=0.9780, Prec=0.5952, Rec=0.8333, F1=0.6944
