import os
import torch

# Seed for reproducibility
torch.manual_seed(42)
SEED = 42

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = device

# Data paths
UNLABELED_DIR     = 'data/train'            # All unlabeled images (normal + anomaly)
SVDD_NORMAL_DIR   = 'data/processed/normal' # Only normal images for one-class SVDD training

# Checkpoint directories
CKPT_DIR          = 'checkpoints/ssl'       # SSL model checkpoints
SVDD_CKPT_DIR     = 'checkpoints/svdd'      # SVDD model checkpoints

# SSL training hyperparameters
BATCH_SIZE        = 64
NUM_WORKERS       = 4
LR                = 1e-5                   # Learning rate for SSL training
MOMENTUM          = 0.9
WEIGHT_DECAY      = 1e-4
EPOCHS            = 10                     # Number of epochs for SSL training
CKPT_FREQ         = 5                      # Save SSL checkpoint every N epochs

# SVDD training hyperparameters
SVDD_LR           = 1e-5                   # Learning rate for SVDD training
SVDD_WEIGHT_DECAY = 1e-6
SVDD_EPOCHS       = 100                    # Number of epochs for SVDD training

# Model configuration
BACKBONE          = 'resnet18'              # Torchvision backbone for feature extraction
INPUT_SIZE        = 227                     # Input resolution (square)

# Normalization constants (ImageNet)
IMG_MEAN          = [0.485, 0.456, 0.406]
IMG_STD           = [0.229, 0.224, 0.225]

# Paths to pretrained checkpoints
SSL_CKPT_PATH     = os.path.join(CKPT_DIR, 'ssl_10.pth')            # SSL final checkpoint
SVDD_CKPT_PATH    = os.path.join(SVDD_CKPT_DIR, 'svdd_epoch100.pth') # SVDD final checkpoint
