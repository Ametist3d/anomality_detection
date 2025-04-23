#!/usr/bin/env python3

import argparse
import os
import shutil
import joblib

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# SVDD-specific imports
from PIL import Image
import torchvision.transforms as T
import torchvision.models as models

# PaDiM-specific imports
import cv2
from trainers.padim import get_backbone, PaDiM
from utils.datasets import CrackDataset
from utils.visualize import overlay_heatmap, plot_score_distribution
import config

#pylint: disable = no-member

@torch.inference_mode()
def load_svdd(ckpt_path):
    class SVDDModel(nn.Module):
        def __init__(self, backbone_name):
            super().__init__()
            backbone = getattr(models, backbone_name)(pretrained=False)
            self.features = nn.Sequential(*list(backbone.children())[:-1])
        def forward(self, x):
            f = self.features(x)
            return f.view(f.size(0), -1)

    device = config.DEVICE
    model = SVDDModel(config.BACKBONE).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    center = ckpt['c'].to(device)
    model.eval()

    preprocess = T.Compose([
        T.Resize(config.INPUT_SIZE + 29),
        T.CenterCrop(config.INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
    ])

    return model, center, preprocess


@torch.inference_mode()
def compute_svdd_distance(model, center, preprocess, path):
    img = Image.open(path).convert('RGB')
    x = preprocess(img).unsqueeze(0).to(config.DEVICE)
    feat = model(x)[0]
    return float(torch.norm(feat - center).item())


def infer_svdd(args):
    model, center, preprocess = load_svdd(args.ckpt)
    # Determine threshold
    if args.threshold is None:
        args.threshold = joblib.load(config.THR_SVDD)

    # Bulk directory sorting
    if args.sort_out:
        root = args.sort_out
        if not os.path.isdir(root):
            raise ValueError("must be a directory when --sort_out is set")

        normal_dir = os.path.join(args.sort_out, 'normal')
        anomaly_dir = os.path.join(args.sort_out, 'anomaly')
        os.makedirs(normal_dir, exist_ok=True)
        os.makedirs(anomaly_dir, exist_ok=True)

        files = [f for f in os.listdir(root)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        results = []

        for fname in tqdm(files, desc="Scoring images", unit="img"):
            path = os.path.join(root, fname)
            dist = compute_svdd_distance(model, center, preprocess, path)
            label = "ANOMALY" if dist > args.threshold else "NORMAL"
            results.append((path, dist, label))

            dest = anomaly_dir if label == "ANOMALY" else normal_dir
            if args.move:
                shutil.move(path, os.path.join(dest, fname))
            else:
                shutil.copy(path, os.path.join(dest, fname))

        # Sort and print
        results.sort(key=lambda x: x[1], reverse=True)
        print("\nSorted results (desc. distance):")
        for path, dist, label in results:
            print(f"{label:7s}  {dist:7.4f}  {path}")
        scores_normal  = [dist for _, dist, label in results if label == "NORMAL"]
        scores_anomaly = [dist for _, dist, label in results if label == "ANOMALY"]
        out_hist = os.path.join(args.sort_out, 'svdd_score_distribution.png')
        plot_score_distribution(scores_normal, scores_anomaly, out_path=out_hist)
        print(f"Saved SVDD score histogram to {out_hist}")
        
    # Single-image mode
    else:
        dist = compute_svdd_distance(model, center, preprocess, args.image)
        label = "ANOMALY" if dist > args.threshold else "NORMAL"
        print(f"Image     : {args.image}")
        print(f"Distance  : {dist:.4f}")
        print(f"Threshold : {args.threshold:.4f}")
        print(f"Label     : {label}")

@torch.inference_mode()
def classify_padim(padim, device, img_path, threshold):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot load {img_path}")
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Extract structural features
    struct = CrackDataset(root_dir='', is_train=True, transform=None).extract_structural(img)

    # Preprocessing
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
    ])
    img_t = transform(img)
    struct_t = transform(struct)
    combined = torch.cat([img_t, struct_t], dim=0).to(device)

    heatmap = padim.score(combined)
    score = float(np.max(heatmap))
    label = "ANOMALY" if score >= threshold else "NORMAL"
    return label, score, heatmap


def infer_padim(args):
    # Determine threshold
    if args.threshold is None:
        args.threshold = joblib.load(config.THR_PADIM)

    # Load PaDiM model
    backbone = get_backbone(config.DEVICE, in_channels=6)
    padim = PaDiM(backbone, pca_components=config.PCA_COMPONENTS, device=config.DEVICE)
    state = torch.load(args.model, map_location=config.DEVICE)
    padim.means = state['means']
    padim.pcas = state['pcas']

    # Bulk directory sorting
    if args.sort_out:
        root = args.sort_out
        if not os.path.isdir(root):
            raise ValueError("must be a directory when --sort_out is set")

        normal_dir = os.path.join(args.sort_out, 'normal')
        anomaly_dir = os.path.join(args.sort_out, 'anomaly')
        os.makedirs(normal_dir, exist_ok=True)
        os.makedirs(anomaly_dir, exist_ok=True)

        scores, labels = [], []
        files = [f for f in os.listdir(root)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for fname in tqdm(files, desc="Classifying & Sorting", unit="img"):
            path = os.path.join(root, fname)
            label, score, _ = classify_padim(padim, config.DEVICE, path, args.threshold)
            scores.append(score)
            labels.append(label)
            dest = normal_dir if label == 'NORMAL' else anomaly_dir
            shutil.move(path, os.path.join(dest, fname))

        # Plot distribution
        plot_score_distribution(
            [s for s, l in zip(scores, labels) if l == 'NORMAL'],
            [s for s, l in zip(scores, labels) if l == 'ANOMALY'],
            out_path=os.path.join(args.sort_out, 'score_distribution.png')
        )
        print(f"Saved score histogram to {os.path.join(args.sort_out, 'score_distribution.png')}")

    # Single-image mode
    else:
        label, score, heatmap = classify_padim(padim, config.DEVICE, args.image, args.threshold)
        print(f"Image         : {args.image}")
        print(f"Anomaly score : {score:.6f}")
        print(f"Threshold     : {args.threshold:.6f}")
        print(f"Prediction    : {label}")
        if args.output_overlay:
            overlay_heatmap(args.image, heatmap, out_path=args.output_overlay)
            print(f"Overlay saved to {args.output_overlay}")
        else:
            overlay_heatmap(args.image, heatmap)


def main():
    parser = argparse.ArgumentParser(description="Unified inference for SVDD and PaDiM")
    parser.add_argument(
        '--mode', choices=['svdd', 'padim'], required=True,
        help="Which inference mode to run"
    )
    parser.add_argument(
        '--image', type=str, 
        help="Path to image file or (with --sort_out) directory of images"
    )
    parser.add_argument(
        '--sort_out', type=str,
        help="Directory in which to sort images into normal/anomaly subfolders"
    )
    parser.add_argument(
        '--threshold', type=float,
        help="Override the anomaly threshold"
    )

    # SVDD-only options
    parser.add_argument(
        '--ckpt', type=str, default=config.SVDD_MODEL,
        help="Path to SVDD checkpoint"
    )
    parser.add_argument(
        '--move', action='store_true',
        help="SVDD: move files when sorting instead of copying"
    )

    # PaDiM-only options
    parser.add_argument(
        '--model', type=str, default=config.PADIM_MODEL,
        help="Path to PaDiM model file (.pt)"
    )
    parser.add_argument(
        '--output_overlay', type=str,
        help="PaDiM: path to save overlay image for single-image mode"
    )

    args = parser.parse_args()

    if args.mode == 'svdd':
        infer_svdd(args)
    else:
        infer_padim(args)

    if args.sort_out:
        root = args.sort_out
        # Bulk mode: process directory `root`
    else:
        img_path = args.image
    # Single-image mode: process `img_path`


if __name__ == '__main__':
    main()
