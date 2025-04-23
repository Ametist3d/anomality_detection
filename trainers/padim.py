import os
import torch
import numpy as np
from torch import nn
from torchvision import models
from sklearn.decomposition import PCA
from tqdm import tqdm
import torchvision.transforms as T
from utils.datasets import CrackDataset
from torch.utils.data import DataLoader

#pylint: disable = no-member

def get_backbone(device, in_channels=6, backbone_name='resnet50'):
    """
    Returns a frozen feature‐extractor (up through the last conv layer) 
    with its first conv adapted to `in_channels`.
    """
    backbone = getattr(models, backbone_name)(pretrained=True)
    original_conv = backbone.conv1
    backbone.conv1 = nn.Conv2d(
        in_channels,
        original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=(original_conv.bias is not None)
    )
    with torch.no_grad():
        # Copy RGB weights into the first 3 channels,
        # and average them into the remaining channels
        backbone.conv1.weight[:, :3]  = original_conv.weight
        backbone.conv1.weight[:, 3:]  = original_conv.weight.mean(dim=1, keepdim=True)
    # Strip off the avgpool & fc layers
    features = nn.Sequential(*list(backbone.children())[:-2])
    return features.to(device)

class PaDiM:
    """
    PaDiM-style anomaly detector:
    - fit() builds a per‐patch Gaussian+PCA on the train set
    - score() computes per‐patch Mahalanobis distances → heatmap
    """

    def __init__(self, backbone: nn.Module, pca_components: int = 100, device: str = 'cpu'):
        self.backbone = backbone.eval()
        self.pca_components = pca_components
        self.device = device
        self.means = {}  # dict[(i,j)] -> mean vector of shape [C]
        self.pcas  = {}  # dict[(i,j)] -> PCA object

    @torch.inference_mode()
    def fit(self, loader):
        """
        Build per‑patch PCA+mean from your *normal* train_loader.
        Uses the backbone’s output spatial dimensions, not the input image size.
        """
        # --- Peek at one batch and run it through the backbone to get C, Hf, Wf
        sample = next(iter(loader))
        img    = sample['image'].to(self.device)       # [B,3,Hin,Win]
        struct = sample['structural'].to(self.device)  # [B,3,Hin,Win]
        x0     = torch.cat([img, struct], dim=1)       # [B,6,Hin,Win]
        fmap0  = self.backbone(x0)                     # [B,C,Hf,Wf]
        _, C, Hf, Wf = fmap0.shape

        # Initialize storage for each patch location
        feats = {(i, j): [] for i in range(Hf) for j in range(Wf)}

        # Accumulate all patch vectors
        for batch in tqdm(loader, desc="Fitting PaDiM", unit="batch", total=len(loader)):
            img    = batch['image'].to(self.device)
            struct = batch['structural'].to(self.device)
            x      = torch.cat([img, struct], dim=1)     # [B,6,Hin,Win]
            fmap   = self.backbone(x)                    # [B,C,Hf,Wf]
            arr    = fmap.permute(0, 2, 3, 1).cpu().numpy()  # [B,Hf,Wf,C]

            B = arr.shape[0]
            for b in range(B):
                for i in range(Hf):
                    for j in range(Wf):
                        feats[(i, j)].append(arr[b, i, j])

        # Fit PCA + compute mean for each (i,j)
        for (i, j), vecs in feats.items():
            mat = np.stack(vecs, axis=0)                   # [N, C]
            pca = PCA(self.pca_components, whiten=True).fit(mat)
            mu  = mat.mean(axis=0)
            self.pcas[(i, j)]  = pca
            self.means[(i, j)] = mu


    @torch.inference_mode()
    def score(self, combined: torch.Tensor):
        """
        combined: Tensor [6, H, W] or [B, 6, H, W]
        Returns:   numpy heatmap [H, W] or [B, H, W]
        """
        # Ensure batch dim
        x = combined.unsqueeze(0) if combined.dim() == 3 else combined  # [B, 6, H, W]
        fmap = self.backbone(x.to(self.device))                         # [B, C, H, W]
        B, C, H, W = fmap.shape
        arr = fmap.permute(0, 2, 3, 1).cpu().numpy()                     # [B, H, W, C]

        dists = np.zeros((B, H, W), dtype=float)
        for b in range(B):
            for i in range(H):
                for j in range(W):
                    vec = arr[b, i, j]                                  # [C]
                    pca = self.pcas[(i, j)]
                    mu  = self.means[(i, j)]
                    # Mahalanobis in PCA subspace
                    z     = pca.transform(vec[None, :])
                    recon = pca.inverse_transform(z).ravel()
                    dists[b, i, j] = np.linalg.norm(vec - recon)

        return dists if B > 1 else dists[0]

def train_padim(
    train_dir: str,
    out_dir: str,
    model_name: str,
    pca_components: int,
    input_size: int,
    img_mean: list,
    img_std: list,
    batch_size: int,
    num_workers: int,
    device: str
):
    """
    Train PaDiM anomaly detector by fitting per-patch Gaussians + PCA.
    """
    # Build data transforms
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=img_mean, std=img_std),
    ])

    # Dataset & loader
    train_ds = CrackDataset(root_dir=train_dir, is_train=True, transform=transform)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    print(f"Loaded {len(train_ds)} images for PaDiM training")

    # Initialize backbone & PaDiM
    backbone = get_backbone(device=device, in_channels=6)
    padim    = PaDiM(backbone, pca_components=pca_components, device=device)

    # Fit (build per-patch PCA+means)
    padim.fit(train_loader)

    # Save model
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'padim_model.pth')
    torch.save({'means': padim.means, 'pcas': padim.pcas}, out_path)
    print(f"PaDiM model saved to {out_path}")