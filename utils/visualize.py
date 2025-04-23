import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

#pylint: disable = no-member

def overlay_heatmap(img_path, heatmap, out_path=None, alpha=0.4):
    """
    Load image, resize heatmap to match image size, overlay, and save/show.
    """
    img = np.array(Image.open(img_path).convert('RGB'))
    heatmap_resized = cv2.resize(
        heatmap,
        (img.shape[1], img.shape[0]),
        interpolation=cv2.INTER_LINEAR
    )
    max_val = heatmap_resized.max() if heatmap_resized.max() > 0 else 1.0
    normed = heatmap_resized / max_val
    h = plt.get_cmap('jet')(normed)[..., :3]
    overlay = (1 - alpha) * (img / 255.0) + alpha * h
    overlay = np.clip(overlay, 0, 1)
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.axis('off')
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()


def plot_score_distribution(scores_normal, scores_anomaly, out_path=None):
    """
    Plot and optionally save a histogram of normal vs. anomaly scores.
    """
    plt.figure(figsize=(6, 4))
    plt.hist([scores_normal, scores_anomaly],
             label=['normal', 'anomaly'],
             bins=30,
             alpha=0.7)
    plt.xlabel('Anomaly Score')
    plt.ylabel('Count')
    plt.title('Score Distribution')
    plt.legend()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()


def plot_curves(y_true, y_scores, out_dir='results'):
    """
    Plot ROC and Precision-Recall curves.
    """
    from sklearn.metrics import roc_curve, precision_recall_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(rec, prec)
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC AUC={roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.title('ROC Curve')
    plt.savefig(f'{out_dir}/roc_curve.png')
    plt.close()
    plt.figure()
    plt.plot(rec, prec, label=f'PR AUC={pr_auc:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.title('Precision-Recall Curve')
    plt.savefig(f'{out_dir}/pr_curve.png')
    plt.close()


def plot_training_loss(losses, out_path, title='Training Loss'):
    """
    Plot and save training loss curve. Always appends .png extension.
    """
    # ensure output directory
    base = os.path.splitext(out_path)[0]
    save_path = base + '.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()
