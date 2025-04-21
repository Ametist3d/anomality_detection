import os
import shutil
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from sklearn.cluster import KMeans
from models.feature_extractor import SVDDModel
import config


def bootstrap_normals(unlabeled_dir, output_dir, n_clusters=2):
    """
    1. Extract deep features via SSL‐pretrained backbone.
    2. Cluster into n_clusters using KMeans.
    3. Identify majority cluster as 'normal'.
    4. Copy those images into output_dir for SVDD training.
    """
    # Prepare transform
    tf = T.Compose([
        T.Resize(config.INPUT_SIZE + 29),
        T.CenterCrop(config.INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(config.IMG_MEAN, config.IMG_STD),
    ])

    # Gather all image paths
    paths = [
        os.path.join(dp, f)
        for dp, dn, files in os.walk(unlabeled_dir)
        for f in files
        if f.lower().endswith(('.png','jpg','jpeg'))
    ]

    # Load SSL weights into feature extractor
    model = SVDDModel(config.BACKBONE).to(config.DEVICE)
    ckpt = torch.load(config.SSL_CKPT_PATH, map_location=config.DEVICE)
    filtered = {k:v for k,v in ckpt.items() if not k.startswith('fc.')}
    model.load_state_dict(filtered, strict=False)
    model.eval()

    # Extract features
    feats = []
    for p in paths:
        img = Image.open(p).convert('RGB')
        x = tf(img).unsqueeze(0).to(config.DEVICE)
        with torch.inference_mode():
            feat = model(x)[0].cpu().numpy()
        feats.append(feat)
    feats = np.vstack(feats)

    # KMeans clustering
    km = KMeans(n_clusters=n_clusters, random_state=config.SEED).fit(feats)
    labels = km.labels_
    # Identify majority cluster
    counts = np.bincount(labels)
    normal_label = counts.argmax()

    # Copy normal cluster images
    os.makedirs(output_dir, exist_ok=True)
    for p, lbl in zip(paths, labels):
        if lbl == normal_label:
            shutil.copy(p, output_dir)

    print(f"Bootstrapped {counts[normal_label]} normal images into {output_dir}")