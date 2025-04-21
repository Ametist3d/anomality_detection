import os
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from models.feature_extractor import SVDDModel
import config


def calibrate(threshold_pct=95):
    """
    Compute mean, std, and specified percentile of SVDD distances on normal images.
    Returns (mu, sigma, tau).
    """
    # Load model & center
    ckpt = torch.load(config.SVDD_CKPT_PATH, map_location=config.DEVICE)
    model = SVDDModel(config.BACKBONE).to(config.DEVICE)
    model.load_state_dict(ckpt['model'])
    center = ckpt['c'].to(config.DEVICE)
    model.eval()

    # Transform pipeline
    tf = T.Compose([
        T.Resize(config.INPUT_SIZE + 29),
        T.CenterCrop(config.INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(config.IMG_MEAN, config.IMG_STD),
    ])

    # Collect distances
    dists = []
    for dp, dn, files in os.walk(config.SVDD_NORMAL_DIR):
        for f in files:
            if not f.lower().endswith(('.png','jpg','jpeg')): continue
            img = Image.open(os.path.join(dp, f)).convert('RGB')
            x = tf(img).unsqueeze(0).to(config.DEVICE)
            with torch.inference_mode():
                feat = model(x)[0]
            dists.append(torch.norm(feat - center).item())
    d = np.array(dists)
    mu, sigma = d.mean(), d.std()
    tau = np.percentile(d, threshold_pct)
    print(f"N={len(d)}  μ={mu:.4f}  σ={sigma:.4f}  {threshold_pct}% τ={tau:.4f}")
    return mu, sigma, tau