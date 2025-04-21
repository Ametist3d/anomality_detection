import os
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as T
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
import config
from models.feature_extractor import SVDDModel


def evaluate_model(ckpt_path, balanced_dir, unbalanced_dir, percentile=95):
    """
    Evaluate SVDD on balanced and unbalanced splits, printing metrics.

    Parameters:
    - ckpt_path: path to SVDD checkpoint (.pth containing 'model' and 'c').
    - balanced_dir: root of balanced test split (with 'normal/' and 'anomaly/' subfolders).
    - unbalanced_dir: root of unbalanced test split.
    - percentile: percentile threshold of normal distances to binarize.
    """
    # Load model and center
    ckpt = torch.load(ckpt_path, map_location=config.DEVICE)
    model = SVDDModel(config.BACKBONE).to(config.DEVICE)
    model.load_state_dict(ckpt['model'])
    center = ckpt['c'].to(config.DEVICE)
    model.eval()

    # Transform
    tf = T.Compose([
        T.Resize(config.INPUT_SIZE + 29),
        T.CenterCrop(config.INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(config.IMG_MEAN, config.IMG_STD),
    ])

    def collect(dist_root):
        distances, labels = [], []
        for label, sub in [(0, 'normal'), (1, 'anomaly')]:
            root = os.path.join(dist_root, sub)
            for dp, dn, files in os.walk(root):
                for f in files:
                    if not f.lower().endswith(('.png','jpg','jpeg')):
                        continue
                    img = Image.open(os.path.join(dp, f)).convert('RGB')
                    x = tf(img).unsqueeze(0).to(config.DEVICE)
                    with torch.inference_mode():
                        feat = model(x)[0]
                    distances.append(torch.norm(feat - center).item())
                    labels.append(label)
        return np.array(distances), np.array(labels)

    # Balanced split
    bd, bl = collect(balanced_dir)
    auc_b = roc_auc_score(bl, bd)
    tau = np.percentile(bd[bl == 0], percentile)
    preds_b = (bd > tau).astype(int)
    print("==== Balanced test split ====")
    print(f"ROC AUC     : {auc_b:.4f}")
    print(f"{percentile}% τ       : {tau:.4f}")
    print(f"Accuracy    : {accuracy_score(bl, preds_b):.4f}")
    print(f"Precision   : {precision_score(bl, preds_b):.4f}")
    print(f"Recall      : {recall_score(bl, preds_b):.4f}")
    print(f"F1 score    : {f1_score(bl, preds_b):.4f}\n")

    # Unbalanced split
    ud, ul = collect(unbalanced_dir)
    auc_u = roc_auc_score(ul, ud)
    preds_u = (ud > tau).astype(int)
    print("==== Unbalanced test split ====")
    print(f"ROC AUC     : {auc_u:.4f}")
    print(f"{percentile}% τ       : {tau:.4f}")
    print(f"Accuracy    : {accuracy_score(ul, preds_u):.4f}")
    print(f"Precision   : {precision_score(ul, preds_u):.4f}")
    print(f"Recall      : {recall_score(ul, preds_u):.4f}")
    print(f"F1 score    : {f1_score(ul, preds_u):.4f}")
