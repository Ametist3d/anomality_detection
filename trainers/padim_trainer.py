import os
import torch
import numpy as np
from torch import nn
from sklearn.decomposition import PCA
from tqdm import tqdm
import torchvision.transforms as T
from utils.datasets import CrackDataset
from torch.utils.data import DataLoader
from utils.feature_extractor import get_feature_backbone as get_backbone

# pylint: disable = no-member


class PaDiM:
    """
    PaDiM-style anomaly detector:
    - fit() builds a per‐patch Gaussian+PCA on the train set
    - score() computes per‐patch Mahalanobis distances → heatmap
    """

    def __init__(
        self, backbone: nn.Module, pca_components: int = 100, device: str = "cpu"
    ):
        self.backbone = backbone.eval()
        self.pca_components = pca_components
        self.device = device
        self.means = {}  # dict[(i,j)] -> mean vector of shape [C]
        self.pcas = {}  # dict[(i,j)] -> PCA object

    @torch.inference_mode()
    def fit(self, loader):
        # Peek at one batch to get spatial dims
        sample = next(iter(loader))
        img = sample["image"].to(self.device)
        struct = sample["structural"].to(self.device)
        x0 = torch.cat([img, struct], dim=1)
        fmap0 = self.backbone(x0)
        _, C, Hf, Wf = fmap0.shape

        # Collect features per patch location
        feats = {(i, j): [] for i in range(Hf) for j in range(Wf)}
        for batch in tqdm(loader, desc="Fitting PaDiM", unit="batch"):
            img = batch["image"].to(self.device)
            struct = batch["structural"].to(self.device)
            x = torch.cat([img, struct], dim=1)
            fmap = self.backbone(x)
            arr = fmap.permute(0, 2, 3, 1).cpu().numpy()  # [B,Hf,Wf,C]
            B = arr.shape[0]
            for b in range(B):
                for i in range(Hf):
                    for j in range(Wf):
                        feats[(i, j)].append(arr[b, i, j])

        # Fit PCA + compute mean for each patch
        for (i, j), vecs in feats.items():
            mat = np.stack(vecs, axis=0)  # [N, C]
            pca = PCA(self.pca_components, whiten=True).fit(mat)
            mu = mat.mean(axis=0)
            self.pcas[(i, j)] = pca
            self.means[(i, j)] = mu

    @torch.inference_mode()
    def score(self, combined: torch.Tensor):
        # combined: [6,H,W] or [B,6,H,W]
        x = combined.unsqueeze(0) if combined.dim() == 3 else combined
        fmap = self.backbone(x.to(self.device))  # [B,C,H,W]
        B, C, H, W = fmap.shape
        arr = fmap.permute(0, 2, 3, 1).cpu().numpy()  # [B,H,W,C]

        dists = np.zeros((B, H, W), dtype=float)
        for b in range(B):
            for i in range(H):
                for j in range(W):
                    vec = arr[b, i, j]
                    pca = self.pcas[(i, j)]
                    mu = self.means[(i, j)]
                    z = pca.transform(vec[None, :])
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
    device: str,
    backbone_name: str,
):
    """
    Train PaDiM anomaly detector by fitting per-patch PCA+Gaussian.
    Saves final model and (optionally) loss/metrics.
    """
    # Data transforms
    transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=img_mean, std=img_std),
        ]
    )

    # DataLoader
    train_ds = CrackDataset(root_dir=train_dir, is_train=True, transform=transform)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    print(f"Loaded {len(train_ds)} images for PaDiM training")

    # Build backbone and PaDiM
    backbone = get_backbone(
        backbone_name=backbone_name, pretrained=True, in_channels=6, strip_avgpool=True
    ).to(device)
    padim = PaDiM(backbone, pca_components=pca_components, device=device)

    # Fit PaDiM model
    padim.fit(train_loader)

    # Save the fitted model
    os.makedirs(out_dir, exist_ok=True)
    out_path = model_name
    torch.save({"means": padim.means, "pcas": padim.pcas}, out_path)
    print(f"PaDiM model saved to {out_path}")
