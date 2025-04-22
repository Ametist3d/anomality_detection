import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

def overlay_heatmap(img_path, heatmap, out_path=None, alpha=0.4):
    """
    Load image, resize heatmap to match image size, overlay, and save/show.
    """
    # Load original image
    img = np.array(Image.open(img_path).convert('RGB'))

    # Resize heatmap to image dimensions
    heatmap_resized = cv2.resize(
        heatmap,
        (img.shape[1], img.shape[0]),
        interpolation=cv2.INTER_LINEAR
    )

    # Normalize resized heatmap (avoid division by zero)
    max_val = heatmap_resized.max() if heatmap_resized.max() > 0 else 1.0
    normed = heatmap_resized / max_val

    # Apply colormap
    h = plt.get_cmap('jet')(normed)[..., :3]

    # Combine image and heatmap
    overlay = (1 - alpha) * (img / 255.0) + alpha * h
    overlay = np.clip(overlay, 0, 1)

    # Display or save
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.axis('off')

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
    roc_auc = auc(fpr,tpr)
    prec, rec, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(rec,prec)

    os.makedirs(out_dir, exist_ok=True)
    # ROC
    plt.figure()
    plt.plot(fpr,tpr,label=f'ROC AUC={roc_auc:.3f}')
    plt.plot([0,1],[0,1],'--',color='gray')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend()
    plt.title('ROC Curve')
    plt.savefig(f'{out_dir}/roc_curve.png')
    plt.close()

    # PR
    plt.figure()
    plt.plot(rec,prec,label=f'PR AUC={pr_auc:.3f}')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.legend()
    plt.title('Precision-Recall Curve')
    plt.savefig(f'{out_dir}/pr_curve.png')
    plt.close()
