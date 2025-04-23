import os
import argparse
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
import config


def build_feature_extractor():
    cache_dir = os.path.expanduser("data/cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TORCH_HOME"] = cache_dir
    torch.hub.set_dir(cache_dir)

    # Load pretrained backbone and strip off the FC layer
    backbone = models.__dict__[config.BACKBONE](pretrained=True)
    feat_extractor = (
        torch.nn.Sequential(*list(backbone.children())[:-1]).to(config.DEVICE).eval()
    )
    return feat_extractor


def compute_centroid_and_threshold(feat_extractor, preprocess):
    # Gather features from normal images
    feats = []
    for fn in os.listdir(config.SVDD_NORMAL_DIR):
        if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        path = os.path.join(config.SVDD_NORMAL_DIR, fn)
        img = Image.open(path).convert("RGB")
        x = preprocess(img).unsqueeze(0).to(config.DEVICE)
        with torch.no_grad():
            v = feat_extractor(x).flatten(1).cpu().numpy()[0]
        feats.append(v)
    feats = np.vstack(feats)  # shape [N, C]

    # Compute centroid and percentile threshold
    centroid = feats.mean(axis=0)
    dists = np.linalg.norm(feats - centroid, axis=1)
    tau = np.percentile(dists, config.THRESHOLD_PERCENTILE)
    print(f"Using {config.THRESHOLD_PERCENTILE}th percentile = {tau:.4f}")
    return centroid, tau


def is_anomaly(path, feat_extractor, preprocess, centroid, tau):
    img = Image.open(path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(config.DEVICE)
    with torch.no_grad():
        v = feat_extractor(x).flatten(1).cpu().numpy()[0]
    d = np.linalg.norm(v - centroid)
    return (d > tau), d


def main():
    parser = argparse.ArgumentParser(
        description="Simple ImageNet‐based anomaly detector"
    )
    parser.add_argument(
        "--image", type=str, required=True, help="Path to the image to test for anomaly"
    )
    args = parser.parse_args()

    # Build feature extractor
    feat_extractor = build_feature_extractor()

    # Define preprocessing transform
    preprocess = transforms.Compose(
        [
            transforms.Resize(config.INPUT_SIZE + 32),
            transforms.CenterCrop(config.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
        ]
    )

    # Compute centroid & threshold from normal data
    centroid, tau = compute_centroid_and_threshold(feat_extractor, preprocess)

    # Run anomaly check on the provided image
    label, dist = is_anomaly(args.image, feat_extractor, preprocess, centroid, tau)
    print(
        f"{'ANOMALY' if label else 'NORMAL'} (distance = {dist:.4f}, threshold = {tau:.4f})"
    )


if __name__ == "__main__":
    main()
