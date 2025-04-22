import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToPILImage, Resize, ToTensor, Normalize

from models.padim import get_backbone, PaDiM
from utils.datasets import CrackDataset
from utils.io import load_config

def main():
    # Load all parameters from YAML (with sensible defaults)
    cfg = load_config('configs/padim.yaml')
    device         = cfg.get('device', 'cpu')
    pca_components = cfg.get('pca_components', 100)
    input_size     = cfg.get('input_size', 227)
    img_mean       = cfg.get('img_mean', [0.485, 0.456, 0.406])
    img_std        = cfg.get('img_std',  [0.229, 0.224, 0.225])
    batch_size     = cfg.get('batch_size', 32)
    num_workers    = cfg.get('num_workers', 4)
    unlabeled_dir  = cfg.get('unlabeled_dir', 'data/train')
    out_dir        = cfg.get('out_dir', 'checkpoints/padim')
    model_name     = cfg.get('model_name', 'padim_model.pt')

    # Build the data transform
    transform = Compose([
        ToPILImage(),
        Resize((input_size, input_size)),
        ToTensor(),
        Normalize(mean=img_mean, std=img_std),
    ])

    # Create dataset & loader
    train_ds     = CrackDataset(root_dir=unlabeled_dir, is_train=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Initialize backbone & PaDiM
    backbone = get_backbone(device=device, in_channels=6)
    padim    = PaDiM(backbone, pca_components=pca_components, device=device)

    # Fit the model (build Gaussians + PCA)
    padim.fit(train_loader)

    # Save the fitted model
    os.makedirs(out_dir, exist_ok=True)
    torch.save({'means': padim.means, 'pcas': padim.pcas},
               os.path.join(out_dir, model_name))

    print(f"PaDiM model saved to {out_dir}/{model_name}")

if __name__ == "__main__":
    main()
