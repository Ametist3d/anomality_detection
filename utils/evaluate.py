import argparse
import os
import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import joblib

import config
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
import cv2

from utils.datasets import get_dataloaders
from trainers.padim_trainer import get_backbone, PaDiM
from utils.visualize import plot_curves, overlay_heatmap, plot_score_distribution

# pylint: disable = no-member


# -- SVDD evaluation --
@torch.inference_mode()
def load_svdd_eval(ckpt_path):
    import torchvision.models as models
    import torch.nn as nn

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
    model.load_state_dict(ckpt["model"])
    center = ckpt["c"].to(device)
    model.eval()
    preprocess = T.Compose(
        [
            T.Resize(config.INPUT_SIZE + 29),
            T.CenterCrop(config.INPUT_SIZE),
            T.ToTensor(),
            T.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
        ]
    )
    return model, center, preprocess


@torch.inference_mode()
def compute_svdd_split(model, center, preprocess, root):
    dists, labels = [], []
    for label, sub in [(0, "normal"), (1, "anomaly")]:
        path = os.path.join(root, sub)
        for dp, _, files in os.walk(path):
            for f in files:
                if not f.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                img = Image.open(os.path.join(dp, f)).convert("RGB")
                x = preprocess(img).unsqueeze(0).to(config.DEVICE)
                feat = model(x)[0]
                d = float(torch.norm(feat - center).item())
                dists.append(d)
                labels.append(label)
    return np.array(dists), np.array(labels)


# -- PaDiM evaluation --
def evaluate_padim_split(padim, loader):
    y_true, y_scores = [], []
    for batch in tqdm(loader, desc="Evaluating PaDiM", unit="batch"):
        imgs = batch["image"]
        structs = batch["structural"]
        labs = batch["label"].numpy()
        for img, struct, lbl in zip(imgs, structs, labs):
            combined = torch.cat([img, struct], dim=0).to(config.DEVICE)
            heatmap = padim.score(combined)
            y_scores.append(float(np.max(heatmap)))
            y_true.append(int(lbl))
    scores = np.array(y_scores)
    y_true = np.array(y_true)
    return scores, y_true


# -- main evaluate --
def main():
    parser = argparse.ArgumentParser(
        description="Unified evaluation for SVDD and PaDiM"
    )
    parser.add_argument("--mode", choices=["svdd", "padim"], required=True)
    parser.add_argument("--balanced", type=str, default=config.BAL_TEST_DIR)
    parser.add_argument("--unbalanced", type=str, default=config.UNBAL_TEST_DIR)
    args = parser.parse_args()

    # load threshold

    if args.mode == "svdd":
        thr = joblib.load(config.THR_SVDD)
        model, center, preprocess = load_svdd_eval(config.SVDD_MODEL)
        for name, root in tqdm(
            [("Balanced", args.balanced), ("Unbalanced", args.unbalanced)]
        ):
            dists, labels = compute_svdd_split(model, center, preprocess, root)
            auc = roc_auc_score(labels, dists)
            preds = (dists >= thr).astype(int)
            acc = accuracy_score(labels, preds)
            prec = precision_score(labels, preds, zero_division=0)
            rec = recall_score(labels, preds, zero_division=0)
            f1 = f1_score(labels, preds, zero_division=0)
            print(
                f"[{name}] AUC={auc:.4f}, Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}"
            )
            # curves
            out_dir = os.path.join("data", "results", "svdd", name.lower())
            os.makedirs(out_dir, exist_ok=True)
            plot_curves(labels, dists, out_dir)
            print(f"Saved SVDD curves to {out_dir}")

    else:
        # PaDiM setup
        backbone = get_backbone(
            backbone_name=config.BACKBONE, pretrained=True, in_channels=6
        ).to(config.DEVICE)
        padim = PaDiM(
            backbone, pca_components=config.PCA_COMPONENTS, device=config.DEVICE
        )
        state = torch.load(config.PADIM_MODEL, map_location=config.DEVICE)
        padim.means, padim.pcas = state["means"], state["pcas"]

        # dataloaders

        ds_cfg = type("c", (), {})()
        ds_cfg.UNLABELED_DIR = config.UNLABELED_DIR
        ds_cfg.BALANCED_TEST_DIR = args.balanced
        ds_cfg.UNBALANCED_TEST_DIR = args.unbalanced
        ds_cfg.INPUT_SIZE = config.INPUT_SIZE
        ds_cfg.IMG_MEAN = config.IMG_MEAN
        ds_cfg.IMG_STD = config.IMG_STD
        ds_cfg.BATCH_SIZE = config.BATCH_SIZE
        ds_cfg.NUM_WORKERS = config.NUM_WORKERS
        _, bal_loader, unbal_loader = get_dataloaders(ds_cfg)

        thr = joblib.load(config.THR_PADIM)

        for name, loader in tqdm(
            [("Balanced", bal_loader), ("Unbalanced", unbal_loader)]
        ):
            scores, labels = evaluate_padim_split(padim, loader)
            auc = roc_auc_score(labels, scores)
            preds = (scores >= thr).astype(int)
            acc = accuracy_score(labels, preds)
            prec = precision_score(labels, preds, zero_division=0)
            rec = recall_score(labels, preds, zero_division=0)
            f1 = f1_score(labels, preds, zero_division=0)
            print(
                f"[{name}] AUC={auc:.4f}, Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}"
            )
            out_dir = os.path.join("data", "results", "padim", name.lower())
            os.makedirs(out_dir, exist_ok=True)
            plot_curves(labels, scores, out_dir)
            print(f"Saved PaDiM curves to {out_dir}")


if __name__ == "__main__":
    main()
