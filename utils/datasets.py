import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from skimage.filters import frangi

class CrackDataset(Dataset):
    """
    Unified Dataset: returns both raw RGB and 3‐channel structural features
    (Canny, vesselness, Gabor) for each image.
    """
    def __init__(self, root_dir, is_train=True, transform=None):
        self.paths = []
        self.labels = []
        self.is_train = is_train
        self.transform = transform

        if is_train:
            # Unlabeled “normal” training images
            for dp, dn, files in os.walk(root_dir):
                for f in files:
                    if f.lower().endswith(('.png','.jpg','.jpeg')):
                        self.paths.append(os.path.join(dp, f))
        else:
            # Labeled test data under root_dir/normal & root_dir/anomaly
            for cls, lab in [('normal',0), ('anomaly',1)]:
                d = os.path.join(root_dir, cls)
                for dp, dn, files in os.walk(d):
                    for f in files:
                        if f.lower().endswith(('.png','.jpg','.jpeg')):
                            self.paths.append(os.path.join(dp,f))
                            self.labels.append(lab)

    def __len__(self):
        return len(self.paths)

    def extract_structural(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        # Canny
        edges = cv2.Canny(enhanced, 50,150)
        # Frangi
        v = frangi(gray.astype(np.float32)/255.0)
        vessel = np.clip((v*255).astype(np.uint8),0,255)
        # Gabor bank → max response
        ks = 31
        greys = gray.astype(np.float32)/255.0
        responses = []
        for θ in np.linspace(0, np.pi, 4, endpoint=False):
            kern = cv2.getGaborKernel((ks,ks),4.0,θ,np.pi/4,0.5,0,ktype=cv2.CV_32F)
            r = cv2.filter2D(greys,cv2.CV_32F,kern)
            responses.append(cv2.normalize(r,None,0,255,cv2.NORM_MINMAX).astype(np.uint8))
        gabor = np.max(np.stack(responses,axis=-1),axis=-1)

        out = np.stack([edges, vessel, gabor], axis=-1)
        return out

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.paths[idx]), cv2.COLOR_BGR2RGB)
        struct = self.extract_structural(img)
        if self.transform:
            img = self.transform(img)
            struct = self.transform(struct)
        if self.is_train:
            return {'image': img, 'structural': struct}
        else:
            return {'image': img, 'structural': struct,
                    'label': torch.tensor(self.labels[idx],dtype=torch.long)}

def get_dataloaders(config):
    tf = T.Compose([
        T.ToPILImage(),
        T.Resize((config.INPUT_SIZE,config.INPUT_SIZE)),
        T.ToTensor(),
        T.Normalize(config.IMG_MEAN, config.IMG_STD)
    ])
    train_ds = CrackDataset(config.UNLABELED_DIR, is_train=True, transform=tf)
    bal_ds   = CrackDataset(config.BALANCED_TEST_DIR, is_train=False, transform=tf)
    unbal_ds = CrackDataset(config.UNBALANCED_TEST_DIR, is_train=False, transform=tf)

    return (
        DataLoader(train_ds,   batch_size=config.BATCH_SIZE, shuffle=True,  num_workers=config.NUM_WORKERS),
        DataLoader(bal_ds,     batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS),
        DataLoader(unbal_ds,   batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS),
    )
