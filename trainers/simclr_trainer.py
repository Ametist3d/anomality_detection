# trainers/ssl_trainer.py
import os
import logging
import random
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

# -- SimCLR Dataset for two augmented views --
class SimCLRDataset(Dataset):
    """
    Unlabeled dataset that returns two augmented views of each image for contrastive learning.
    """
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.paths = [
            os.path.join(dp, f)
            for dp, dn, files in os.walk(root_dir)
            for f in files
            if f.lower().endswith(('.png','.jpg','.jpeg'))
        ]
        if not self.paths:
            raise RuntimeError(f"No images found in {root_dir}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        # Generate two views
        xi = self.transform(img)
        xj = self.transform(img)
        return xi, xj

# -- NT-Xent Loss --
class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.mask = self._get_correlated_mask().to(device)

    def _get_correlated_mask(self):
        N = 2 * self.batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask.fill_diagonal_(0)
        for i in range(self.batch_size):
            mask[i, i + self.batch_size] = 0
            mask[i + self.batch_size, i] = 0
        return mask

    def forward(self, zis, zjs):
        """Compute NT-Xent loss for a batch of projections zis and zjs"""
        representations = torch.cat([zis, zjs], dim=0)  # [2B, D]
        # similarity matrix
        sim = torch.matmul(representations, representations.T) / self.temperature
        # remove self
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        positives = torch.cat([sim_i_j, sim_j_i], dim=0)

        # mask out positives
        mask = self.mask
        negatives = sim[mask].view(2 * self.batch_size, -1)

        # logits and labels
        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
        labels = torch.zeros(2 * self.batch_size, dtype=torch.long).to(self.device)

        loss = self.criterion(logits, labels)
        loss = loss / (2 * self.batch_size)
        return loss

# -- SimCLR training loop --
def train_simclr():
    # Data augmentations for SimCLR
    sim_transform = T.Compose([
        T.RandomResizedCrop(config.INPUT_SIZE),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.GaussianBlur(kernel_size=3),
        T.ToTensor(),
        T.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD)
    ])

    dataset = SimCLRDataset(config.UNLABELED_DIR, transform=sim_transform)
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        drop_last=True
    )
    logger.info(f"Loaded {len(dataset)} images for SimCLR SSL pretraining")

    # Backbone + projection head
    backbone = getattr(models, config.BACKBONE)(pretrained=False)
    num_ftrs = backbone.fc.in_features
    backbone.fc = nn.Identity()
    projector = nn.Sequential(
        nn.Linear(num_ftrs, num_ftrs),
        nn.ReLU(),
        nn.Linear(num_ftrs, num_ftrs)
    )
    model = nn.Sequential(backbone, projector).to(config.DEVICE)

    # Loss and optimizer
    criterion = NTXentLoss(batch_size=config.BATCH_SIZE,
                           temperature=0.5,
                           device=config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LR,
                           weight_decay=config.WEIGHT_DECAY)

    # Training loop
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xi, xj in loader:
            xi = xi.to(config.DEVICE)
            xj = xj.to(config.DEVICE)
            zi = model(xi)
            zj = model(xj)
            loss = criterion(zi, zj)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        logger.info(f"Epoch [{epoch}/{config.EPOCHS}] SimCLR Loss: {avg_loss:.4f}")

        # Checkpoint every ckpt_freq
        if epoch % config.CKPT_FREQ == 0 or epoch == config.EPOCHS:
            os.makedirs(config.CKPT_DIR, exist_ok=True)
            path = os.path.join(config.CKPT_DIR, f'simclr_{epoch}.pth')
            torch.save(model.state_dict(), path)
            logger.info(f"Saved SimCLR checkpoint: {path}")

if __name__ == '__main__':
    train_ssl()
