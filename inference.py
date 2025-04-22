import os
import argparse
import torch
import cv2
import joblib
import numpy as np
from torchvision import transforms as T

from utils.visualize import overlay_heatmap
from models.padim import get_backbone, PaDiM
from utils.datasets import CrackDataset
from utils.io import load_config   # use your io.py helper

def main():
    parser = argparse.ArgumentParser(description="PaDiM single‑image inference")
    parser.add_argument('--image',          type=str,   required=True,  help="Path to input image")
    parser.add_argument('--model',          type=str,   default='checkpoints/padim/padim_model.pt',
                        help="Path to saved PaDiM model (.pt)")
    parser.add_argument('--config',         type=str,   default='configs/padim.yaml',
                        help="Path to PaDiM YAML config")
    parser.add_argument('--threshold',      type=float, required=False,
                        help="Absolute anomaly‑score threshold for classification")
    parser.add_argument('--output_overlay', type=str,   default=None,
                        help="Where to save the overlay (optional)")
    args = parser.parse_args()

    # --- load YAML config (no more import config)
    cfg = load_config(args.config)
    device    = cfg['device']
    input_sz  = cfg['input_size']     # e.g. 227
    img_mean  = cfg['img_mean']       # e.g. [0.485,0.456,0.406]
    img_std   = cfg['img_std']        # e.g. [0.229,0.224,0.225]

    if args.threshold is None:
        args.threshold = joblib.load(os.path.join(cfg['out_dir'], 'best_threshold.pkl'))

    # --- build model
    backbone = get_backbone(device, in_channels=6)
    padim    = PaDiM(backbone,
                     pca_components=cfg['pca_components'],
                     device=device)

    state = torch.load(args.model, map_location=device)
    padim.means = state['means']
    padim.pcas  = state['pcas']

    # --- read & preprocess image
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot load {args.image}")
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    struct = CrackDataset(root_dir='', is_train=True, transform=None).extract_structural(img)

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((input_sz, input_sz)),
        T.ToTensor(),
        T.Normalize(mean=img_mean, std=img_std),
    ])
    img_t    = transform(img)
    struct_t = transform(struct)
    combined = torch.cat([img_t, struct_t], dim=0).to(device)  # [6, H, W]

    # --- anomaly scoring
    heatmap = padim.score(combined)       
    score   = float(np.max(heatmap))
    label   = "ANOMALY" if score >= args.threshold else "NORMAL"

    print(f"Image         : {args.image}")
    print(f"Anomaly score : {score:.6f}")
    print(f"Threshold     : {args.threshold:.6f}")
    print(f"Prediction    : {label}")

    # --- optional overlay
    if args.output_overlay:
        overlay_heatmap(args.image, heatmap, out_path=args.output_overlay)
        print(f"Overlay saved to {args.output_overlay}")
    else:
        overlay_heatmap(args.image, heatmap)

if __name__ == "__main__":
    main()
