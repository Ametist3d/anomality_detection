import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import joblib

from models.padim import get_backbone, PaDiM
from utils.datasets import get_dataloaders
from utils.io import load_config  

def load_padim(cfg, device):
    data = torch.load(os.path.join(cfg['out_dir'], cfg['model_name']), map_location=device)
    backbone = get_backbone(device, in_channels=6)
    padim = PaDiM(backbone, pca_components=cfg['pca_components'], device=device)
    padim.means = data['means']
    padim.pcas  = data['pcas']
    return padim

def get_best_threshold(scores, y_true):
    """
    Return the threshold that maximizes F1 on (scores, y_true).
    """
    cands = np.linspace(scores.min(), scores.max(), 200)
    f1s   = [f1_score(y_true, (scores >= t).astype(int), zero_division=0) for t in cands]
    best_i = np.argmax(f1s)
    return cands[best_i]

def main():
    # Load evaluation config from YAML
    cfg = load_config('configs/padim.yaml')
    device = cfg['device']

    # Build a simple config‐like object for dataloaders
    ds_cfg = type('dc', (), {})()
    ds_cfg.UNLABELED_DIR       = cfg.get('unlabeled_dir',       'data/train')
    ds_cfg.BALANCED_TEST_DIR   = cfg.get('balanced_test_dir',   'data/test_balanced')
    ds_cfg.UNBALANCED_TEST_DIR = cfg.get('unbalanced_test_dir', 'data/test_unbalanced')
    ds_cfg.INPUT_SIZE          = cfg.get('input_size',          227)
    ds_cfg.IMG_MEAN            = cfg.get('img_mean',            [0.485, 0.456, 0.406])
    ds_cfg.IMG_STD             = cfg.get('img_std',             [0.229, 0.224, 0.225])
    ds_cfg.BATCH_SIZE          = cfg.get('batch_size',          32)
    ds_cfg.NUM_WORKERS         = cfg.get('num_workers',         4)

    # Get test loaders
    _, bal_loader, unbal_loader = get_dataloaders(ds_cfg)

    # Load PaDiM model
    padim = load_padim(cfg, device)

    thresholds = []
    for name, loader in [('Balanced', bal_loader), ('Unbalanced', unbal_loader)]:
        y_true, y_scores = [], []

        for batch in tqdm(loader, desc=f"Evaluating {name}", unit="batch", total=len(loader)):
            imgs    = batch['image']
            structs = batch['structural']
            labels  = batch['label'].numpy()

            for img, struct, lbl in zip(imgs, structs, labels):
                combined = torch.cat([img, struct], dim=0).to(device)
                heatmap  = padim.score(combined)
                y_scores.append(float(np.max(heatmap)))
                y_true.append(int(lbl))

        scores = np.array(y_scores)
        auc    = roc_auc_score(y_true, scores)

        thr_pct = np.percentile(scores[np.array(y_true) == 0], cfg['threshold_percentile'])
        preds   = (scores >= thr_pct).astype(int)

        acc  = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        rec  = recall_score(y_true, preds, zero_division=0)
        f1   = f1_score(y_true, preds, zero_division=0)

        print(f"[{name}] AUC={auc:.4f}, Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")

        # compute and store split’s optimal F1 threshold
        best_thr = get_best_threshold(scores, y_true)
        print(f"[{name}] optimal F1 threshold: {best_thr:.4f}")
        thresholds.append(best_thr)

    # average threshold across both splits
    avg_thr = float(np.mean(thresholds))
    print(f"Average optimal threshold: {avg_thr:.4f}")

    # save average threshold for inference
    os.makedirs(cfg['out_dir'], exist_ok=True)
    joblib.dump(avg_thr, os.path.join(cfg['out_dir'], 'best_threshold.pkl'))
    print(f"Saved average threshold to {os.path.join(cfg['out_dir'], 'best_threshold.pkl')}")

if __name__ == "__main__":
    main()
