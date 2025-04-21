import os
import logging
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import config
from models.feature_extractor import SVDDModel

# configure logging and reproducibility
torch.manual_seed(config.SEED)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class NormalDataset(Dataset):
    """
    Loads only normal (non-anomalous) images for one-class SVDD training.
    Recursively scans `SVDD_NORMAL_DIR`.
    """
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.files = [
            os.path.join(dp, f)
            for dp, dn, files in os.walk(root_dir)
            for f in files
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        if not self.files:
            raise RuntimeError(f"No images found in normal directory: {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


def train_svdd():
    # transforms match SSL stage
    transform = T.Compose([
        T.Resize(config.INPUT_SIZE + 29),
        T.CenterCrop(config.INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
    ])

    # dataset + loader
    dataset = NormalDataset(config.SVDD_NORMAL_DIR, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    logger.info(f"Loaded {len(dataset)} normal samples for SVDD training")

    # init model and optionally load SSL-pretrained backbone weights
    model = SVDDModel(config.BACKBONE).to(config.DEVICE)
    if os.path.exists(config.SSL_CKPT_PATH):
        ssl_ckpt = torch.load(config.SSL_CKPT_PATH, map_location=config.DEVICE)
        # filter out rotation head parameters (fc.*)
        pretrained = {k: v for k, v in ssl_ckpt.items() if not k.startswith('fc.')}
        model.load_state_dict(pretrained, strict=False)
        logger.info(f"Loaded SSL weights from {config.SSL_CKPT_PATH}")

    # compute center c by averaging features of all normals
    model.eval()
    with torch.inference_mode():
        feats = []
        for imgs in DataLoader(dataset, batch_size=config.BATCH_SIZE, num_workers=0):
            imgs = imgs.to(config.DEVICE)
            z = model(imgs)
            feats.append(z.cpu())
        feats = torch.cat(feats, dim=0)
    center = feats.mean(dim=0).to(config.DEVICE)
    logger.info("Initialized SVDD center (c)")

    # optimizer for one-class loss
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.SVDD_LR,
        weight_decay=config.SVDD_WEIGHT_DECAY
    )

    # training loop: minimize ||f(x) - c||^2
    for epoch in range(1, config.SVDD_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for imgs in loader:
            imgs = imgs.to(config.DEVICE)
            z = model(imgs)
            loss = torch.mean((z - center) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)

        avg_loss = total_loss / len(dataset)
        logger.info(f"Epoch [{epoch}/{config.SVDD_EPOCHS}] SVDD Loss: {avg_loss:.6f}")

        # checkpoint
        if epoch % config.CKPT_FREQ == 0 or epoch == config.SVDD_EPOCHS:
            os.makedirs(config.SVDD_CKPT_DIR, exist_ok=True)
            path = os.path.join(config.SVDD_CKPT_DIR, f"svdd_epoch{epoch}.pth")
            torch.save({'model': model.state_dict(), 'c': center}, path)
            logger.info(f"Saved SVDD checkpoint: {path}")


if __name__ == "__main__":
    train_svdd()
