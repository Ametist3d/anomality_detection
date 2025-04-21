import os
import random
import logging
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
import config

# Configure logging and seed
torch.manual_seed(config.SEED)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Rotation labels for self-supervision
ROTATIONS = [0, 90, 180, 270]

class RotationDataset(Dataset):
    """
    Dataset for rotation-based self-supervised learning on unlabeled images.
    Recursively scans `root_dir` for images and assigns a random rotation label.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        # Recursively gather all image files
        self.filenames = [
            os.path.join(dp, f)
            for dp, dn, files in os.walk(root_dir)
            for f in files
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        if not self.filenames:
            raise RuntimeError(f"No images found in {root_dir}")
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = Image.open(self.filenames[idx]).convert('RGB')
        # Assign a random rotation label
        label = random.randint(0, len(ROTATIONS) - 1)
        angle = ROTATIONS[label]
        img = img.rotate(angle)
        if self.transform:
            img = self.transform(img)
        return img, label


def train_ssl():
    # Prepare image transforms
    transform = T.Compose([
        T.Resize(config.INPUT_SIZE + 29),  # e.g. 256 for cropping to INPUT_SIZE
        T.CenterCrop(config.INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD)
    ])

    # Create dataset and dataloader
    dataset = RotationDataset(root_dir=config.UNLABELED_DIR, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    logger.info(f"Loaded {len(dataset)} unlabeled images from {config.UNLABELED_DIR}")

    # Initialize model (pretrained backbone + rotation head)
    model = getattr(models, config.BACKBONE)(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(ROTATIONS))
    model = model.to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.LR,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY
    )

    # Training loop
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for imgs, labels in loader:
            imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        epoch_loss = running_loss / len(dataset)
        logger.info(f"Epoch [{epoch}/{config.EPOCHS}] Loss: {epoch_loss:.4f}")

        # Save checkpoint at intervals
        if epoch % config.CKPT_FREQ == 0 or epoch == config.EPOCHS:
            os.makedirs(config.CKPT_DIR, exist_ok=True)
            ckpt_path = os.path.join(config.CKPT_DIR, f'ssl_{epoch}.pth')
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")

if __name__ == '__main__':
    train_ssl()