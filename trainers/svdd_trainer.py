import os
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
import torchvision.models as models
from utils.visualize import plot_training_loss
import logging

# pylint: disable = no-member

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

torch.manual_seed(42)


class NormalDataset(Dataset):
    """
    Dataset that loads only normal (non-anomalous) images for one-class SVDD training.
    """

    def __init__(self, root_dir: str, transform=None):
        self.files = [
            os.path.join(dp, f)
            for dp, dn, files in os.walk(root_dir)
            for f in files
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if not self.files:
            raise RuntimeError(f"No images found in normal directory {root_dir}")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


class SVDDModel(nn.Module):
    """
    Deep SVDD feature extractor (backbone without classification head).
    """

    def __init__(self, backbone_name: str):
        super().__init__()
        backbone = getattr(models, backbone_name)(pretrained=False)
        self.features = nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, x):
        f = self.features(x)
        return f.view(f.size(0), -1)


def train_svdd(
    normal_dir: str,
    batch_size: int,
    num_workers: int,
    lr: float,
    weight_decay: float,
    epochs: int,
    ckpt_dir: str,
    ssl_ckpt_path: Optional[str],
    backbone_name: str,
    device: str,
):
    """
    One-class SVDD training on normal images. Saves final checkpoint and loss curve plot.
    """
    # Import config constants
    from config import INPUT_SIZE, IMG_MEAN, IMG_STD

    # Preprocessing transform
    transform = T.Compose(
        [
            T.Resize(INPUT_SIZE + 29),
            T.CenterCrop(INPUT_SIZE),
            T.ToTensor(),
            T.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ]
    )

    # Dataset and loader
    dataset = NormalDataset(normal_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    logger.info(f"Loaded {len(dataset)} normal samples from {normal_dir}")

    # Initialize model and optional SSL weights
    model = SVDDModel(backbone_name).to(device)
    if ssl_ckpt_path and os.path.exists(ssl_ckpt_path):
        ssl_weights = torch.load(ssl_ckpt_path, map_location=device)
        filtered = {k: v for k, v in ssl_weights.items() if not k.startswith("fc.")}
        model.load_state_dict(filtered, strict=False)
        logger.info(f"Loaded SSL weights from {ssl_ckpt_path}")

    # Compute hypersphere center c
    model.eval()
    with torch.inference_mode():
        feats = []
        for batch in DataLoader(dataset, batch_size=batch_size, num_workers=0):
            feats.append(model(batch.to(device)))
        feats = torch.cat(feats, dim=0)
    c = feats.mean(dim=0).to(device)
    logger.info("Initialized SVDD center c")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop
    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs in tqdm(loader, desc=f"SVDD Epoch {epoch}/{epochs}", unit="batch"):
            imgs = imgs.to(device)
            feats = model(imgs)
            loss = torch.mean((feats - c) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
        avg_loss = total_loss / len(dataset)
        losses.append(avg_loss)
        logger.info(f"SVDD Epoch [{epoch}/{epochs}] Loss: {avg_loss:.6f}")

    # Save final checkpoint
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "svdd_model.pth")
    torch.save({"model": model.state_dict(), "c": c}, ckpt_path)
    logger.info(f"Saved final SVDD checkpoint: {ckpt_path}")

    # Plot and save loss curve
    loss_plot = os.path.join(ckpt_dir, "svdd_loss_curve")
    plot_training_loss(losses, loss_plot, title="SVDD Training Loss")
    logger.info(f"Saved SVDD loss curve to {loss_plot}.png")
