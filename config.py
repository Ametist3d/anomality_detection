import os
import torch

# ─── Custom cache for pretrained models ────────────────────────────────────────
TORCH_CACHE_DIR = os.path.expanduser("data/cache")  # or whatever path you choose
os.makedirs(TORCH_CACHE_DIR, exist_ok=True)

# Tell both torch.hub and torchvision’s model_zoo to use it
os.environ["TORCH_HOME"] = TORCH_CACHE_DIR
torch.hub.set_dir(TORCH_CACHE_DIR)

# ─── Device & Seed ─────────────────────────────────────────────────────────────
torch.manual_seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# ─── General Data Paths ─────────────────────────────────────────────────────────
UNLABELED_DIR = "data/train"  # SSL & PaDiM unlabeled
SVDD_NORMAL_DIR = "data/train/normal"  # SVDD one‐class data
BAL_TEST_DIR = "data/test_balanced"
UNBAL_TEST_DIR = "data/test_unbalanced"
TEST_DIR = "data/test"

# ─── Checkpoint & Output ───────────────────────────────────────────────────────
SSL_CKPT_DIR = "checkpoints/ssl"
SSL_MODEL = os.path.join(SSL_CKPT_DIR, "ssl_model.pth")
SVDD_CKPT_DIR = "checkpoints/svdd"
SVDD_MODEL = os.path.join(SVDD_CKPT_DIR, "svdd_model.pth")
PADIM_CKPT_DIR = "checkpoints/padim"
PADIM_MODEL = os.path.join(PADIM_CKPT_DIR, "padim_model.pt")

THR_SVDD = "checkpoints/svdd_thr.pkl"
THR_PADIM = "checkpoints/padim_thr.pkl"

# ─── Shared Hyperparameters ────────────────────────────────────────────────────
BACKBONE = "resnet18"
BATCH_SIZE = 64
NUM_WORKERS = 4
INPUT_SIZE = 227
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

# ─── SSL Training ──────────────────────────────────────────────────────────────
LR = 1e-5
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
EPOCHS_SSL = 10
SSL_CKPT_FREQ = 5

# ─── SVDD Training ─────────────────────────────────────────────────────────────
SVDD_LR = 1e-5
SVDD_WEIGHT_DECAY = 1e-6
EPOCHS_SVDD = 100
SVDD_CKPT_FREQ = 20

# ─── PaDiM Training ────────────────────────────────────────────────────────────
PCA_COMPONENTS = 100
THRESHOLD_PERCENTILE = 95
