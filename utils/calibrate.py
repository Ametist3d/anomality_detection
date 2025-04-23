import argparse
import os
import torch
import numpy as np
import joblib
from PIL import Image
import torchvision.transforms as T
import cv2

import config
from tqdm import tqdm
from trainers.svdd_trainer import SVDDModel
from utils.datasets import CrackDataset
from trainers.padim import get_backbone, PaDiM

# pylint: disable = no-member


@torch.inference_mode()
def load_svdd_model():

    device = config.DEVICE
    model = SVDDModel(config.BACKBONE).to(device)
    ckpt = torch.load(config.SVDD_MODEL, map_location=device)
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
def compute_svdd_dists(model, center, preprocess):
    dists = []
    for dp, _, files in tqdm(os.walk(config.SVDD_NORMAL_DIR)):
        for f in files:
            if not f.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            img = Image.open(os.path.join(dp, f)).convert("RGB")
            x = preprocess(img).unsqueeze(0).to(config.DEVICE)
            dists.append(float(torch.norm(model(x)[0] - center).item()))
    return np.array(dists)


def calibrate_svdd(percentile, output_path):
    model, center, preprocess = load_svdd_model()
    dists = compute_svdd_dists(model, center, preprocess)
    thr = np.percentile(dists, percentile)
    print(f"SVDD {percentile}th percentile = {thr:.4f}")
    joblib.dump(thr, output_path)
    print(f"Saved SVDD threshold to {output_path}")


def calibrate_padim(percentile, output_path):
    # Load PaDiM
    backbone = get_backbone(config.DEVICE, in_channels=6)
    padim = PaDiM(backbone, pca_components=config.PCA_COMPONENTS, device=config.DEVICE)
    state = torch.load(config.PADIM_MODEL, map_location=config.DEVICE)
    padim.means, padim.pcas = state["means"], state["pcas"]

    # Compute normal scores
    scores = []
    for dp, _, files in os.walk(config.TEST_DIR):
        for f in tqdm(files):
            if not f.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            path = os.path.join(dp, f)
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            struct = CrackDataset(
                root_dir="", is_train=True, transform=None
            ).extract_structural(img)
            trans = T.Compose(
                [
                    T.ToPILImage(),
                    T.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
                    T.ToTensor(),
                    T.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
                ]
            )
            img_t, struct_t = trans(img), trans(struct)
            combined = torch.cat([img_t, struct_t], dim=0).to(config.DEVICE)
            heatmap = padim.score(combined)
            scores.append(float(np.max(heatmap)))

    thr = np.percentile(np.array(scores), percentile)
    print(f"PaDiM {percentile}th percentile = {thr:.4f}")
    joblib.dump(thr, output_path)
    print(f"Saved PaDiM threshold to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Calibrate anomaly‐score thresholds")
    parser.add_argument(
        "--mode",
        choices=["svdd", "padim"],
        required=True,
        help="Which model to calibrate",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=95,
        help="Percentile of normal scores to use as threshold",
    )
    args = parser.parse_args()

    # Always use config paths internally
    if args.mode == "svdd":
        calibrate_svdd(args.percentile, config.THR_SVDD)
    else:
        calibrate_padim(args.percentile, config.THR_PADIM)


if __name__ == "__main__":
    main()
