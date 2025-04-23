import argparse
import config

# SSL training function
from trainers.ssl_trainer import train_ssl

# SVDD training function
from trainers.svdd_trainer import train_svdd

# PaDiM components
from trainers.padim_trainer import train_padim

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified trainer for SSL, SVDD, and PaDiM"
    )
    parser.add_argument(
        "--mode",
        choices=["ssl", "svdd", "padim"],
        required=True,
        help="Which model to train",
    )
    parser.add_argument(
        "--epochs", type=int, help="Override number of epochs for SSL or SVDD"
    )
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument(
        "--lr", type=float, help="Override learning rate for SSL or SVDD"
    )
    args = parser.parse_args()

    # Determine overrides
    batch_size = args.batch_size or config.BATCH_SIZE

    if args.mode == "ssl":
        # SSL training
        epochs = args.epochs or config.EPOCHS_SSL
        lr = args.lr or config.LR
        train_ssl(
            unlabeled_dir=config.UNLABELED_DIR,
            batch_size=batch_size,
            num_workers=config.NUM_WORKERS,
            lr=args.lr or config.LR,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY,
            epochs=args.epochs or config.EPOCHS_SSL,
            ckpt_dir=config.SSL_CKPT_DIR,
            device=config.DEVICE,
            backbone_name=config.BACKBONE,
        )

    elif args.mode == "svdd":
        # SVDD training
        epochs = args.epochs or config.EPOCHS_SVDD
        lr = args.lr or config.SVDD_LR
        train_svdd(
            normal_dir=config.SVDD_NORMAL_DIR,
            batch_size=batch_size,
            num_workers=config.NUM_WORKERS,
            lr=lr,
            weight_decay=config.SVDD_WEIGHT_DECAY,
            epochs=epochs,
            ckpt_dir=config.SVDD_CKPT_DIR,
            ssl_ckpt_path=getattr(config, "SSL_CKPT_PATH", None),
            device=config.DEVICE,
            backbone_name=config.BACKBONE,
        )

    else:
        # PaDiM training
        train_padim(
            train_dir=config.UNLABELED_DIR,
            out_dir=config.PADIM_CKPT_DIR,
            model_name=config.PADIM_MODEL,
            pca_components=config.PCA_COMPONENTS,
            input_size=config.INPUT_SIZE,
            img_mean=config.IMG_MEAN,
            img_std=config.IMG_STD,
            batch_size=batch_size,
            num_workers=config.NUM_WORKERS,
            device=config.DEVICE,
            backbone_name=config.BACKBONE
        )
