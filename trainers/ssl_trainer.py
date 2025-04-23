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
from tqdm import tqdm
from utils.visualize import plot_training_loss

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)
torch.manual_seed(42)

# Rotation labels for self-supervision
ROTATIONS = [0, 90, 180, 270]

class RotationDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
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
        label = random.randint(0, len(ROTATIONS) - 1)
        angle = ROTATIONS[label]
        img = img.rotate(angle)
        if self.transform:
            img = self.transform(img)
        return img, label


def train_ssl(
    unlabeled_dir: str,
    batch_size: int,
    num_workers: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    epochs: int,
    ckpt_dir: str,
    ckpt_freq: int,
    device: str,
    backbone_name: str
):
    """
    Self-supervised rotation prediction on unlabeled data.
    Saves model checkpoints and a loss curve plot.
    """
    # Prepare transforms
    from config import INPUT_SIZE, IMG_MEAN, IMG_STD
    transform = T.Compose([
        T.Resize(INPUT_SIZE + 29),
        T.CenterCrop(INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(mean=IMG_MEAN, std=IMG_STD)
    ])

    # DataLoader
    dataset = RotationDataset(unlabeled_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    logger.info(f"Loaded {len(dataset)} unlabeled images from {unlabeled_dir}")

    # Model and optimizer
    model = getattr(models, backbone_name)(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(ROTATIONS))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )


    # Training loop
    losses = []
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for imgs, labels in tqdm(loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        epoch_loss = running_loss / len(dataset)
        losses.append(epoch_loss)
        logger.info(f"SSL Epoch [{epoch}/{epochs}] Loss: {epoch_loss:.4f}")

    # Final checkpoint
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'ssl_model.pth')
    torch.save(model.state_dict(), ckpt_path)
    logger.info(f"Saved final SSL checkpoint: {ckpt_path}")

    # Plot loss curve
    loss_plot = os.path.join(ckpt_dir, 'ssl_loss_curve')
    plot_training_loss(losses, loss_plot, title='SSL Training Loss')
    logger.info(f"Saved SSL loss curve to {loss_plot}.png")
