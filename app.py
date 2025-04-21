import argparse
from PIL import Image
import torch
import torchvision.transforms as T

import config
from models.feature_extractor import SVDDModel
from utils.bootstrap import bootstrap_normals
from utils.calibration import calibrate
from utils.evaluate import evaluate_model
from trainers import simclr_trainer, ssl_trainer, svdd_trainer


def main():
    parser = argparse.ArgumentParser(description="Anomaly Detection Pipeline")
    parser.add_argument(
        '--stage', type=str, required=True,
        choices=['bootstrap','ssl','simclr','svdd','calibration','evaluate','detect'],
        help='Pipeline stage to run'
    )
    parser.add_argument('--image', type=str, help='Image path for detect stage')
    parser.add_argument('--threshold', type=float, help='Distance threshold for detect stage')
    parser.add_argument('--balanced_dir', type=str, default='data/test_balanced', help='Balanced test directory')
    parser.add_argument('--unbalanced_dir', type=str, default='data/test_unbalanced', help='Unbalanced test directory')
    args = parser.parse_args()

    if args.stage == 'bootstrap':
        print(f"Bootstrapping normals from {config.UNLABELED_DIR} → {config.SVDD_NORMAL_DIR}")
        bootstrap_normals(config.UNLABELED_DIR, config.SVDD_NORMAL_DIR)
    elif args.stage == 'ssl':
        ssl_trainer.train_ssl()
    elif args.stage == 'simclr':
        simclr_trainer.train_simclr()
    elif args.stage == 'svdd':
        svdd_trainer.train_svdd()
    elif args.stage == 'calibration':
        calibrate()
    elif args.stage == 'evaluate':
        evaluate_model(
            ckpt_path=config.SVDD_CKPT_PATH,
            balanced_dir=args.balanced_dir,
            unbalanced_dir=args.unbalanced_dir
        )
    elif args.stage == 'detect':
        if not args.image or args.threshold is None:
            parser.error('detect stage requires --image and --threshold')
        # load model & center
        ckpt = torch.load(config.SVDD_CKPT_PATH, map_location=config.DEVICE)
        model = SVDDModel(config.BACKBONE).to(config.DEVICE)
        model.load_state_dict(ckpt['model'])
        center = ckpt['c'].to(config.DEVICE)
        model.eval()

        tf = T.Compose([
            T.Resize(config.INPUT_SIZE + 29),
            T.CenterCrop(config.INPUT_SIZE),
            T.ToTensor(),
            T.Normalize(config.IMG_MEAN, config.IMG_STD),
        ])
        img = Image.open(args.image).convert('RGB')
        x = tf(img).unsqueeze(0).to(config.DEVICE)
        with torch.inference_mode():
            feat = model(x)[0]
        dist = torch.norm(feat - center).item()
        label = 'anomaly' if dist > args.threshold else 'normal'
        print(f"Distance={dist:.4f}  Threshold={args.threshold:.4f}  Label={label}")

if __name__ == '__main__':
    main()
