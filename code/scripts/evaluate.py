"""
evaluate.py — Run inference on test set or unseen domains
─────────────────────────────────────────────────────────
Usage:
    python scripts/evaluate.py --config configs/default.yaml \
        --checkpoint checkpoints/best.pth

    # Or specify a custom test directory:
    python scripts/evaluate.py --config configs/default.yaml \
        --checkpoint checkpoints/best.pth --test_dir ./unseen_domain
"""

import argparse
import sys
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import LeafDiseaseDataset
from data.transforms import get_val_transform
from models.resnet18_unet import ResNet18UNet
from utils.helpers import load_checkpoint
from utils.metrics import compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_dir", type=str, default=None,
                        help="Override test directory (default: from config)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dir = args.test_dir or cfg["data"]["test_dir"]

    # ── Model ────────────────────────────────
    mcfg = cfg["model"]
    model = ResNet18UNet(
        num_classes=mcfg["num_classes"],
        pretrained=False,
        in_channels=mcfg["in_channels"],
        use_aspp=mcfg.get("use_aspp", True),
        aspp_rates=tuple(mcfg.get("aspp_rates", [6, 12, 18])),
    ).to(device)

    load_checkpoint(model, args.checkpoint)
    model.eval()

    # ── Test dataset ─────────────────────────
    test_ds = LeafDiseaseDataset(
        root_dir=test_dir,
        image_size=cfg["data"]["image_size"],
        transform=get_val_transform(cfg["data"]["image_size"]),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
    )

    # ── Inference ────────────────────────────
    all_preds, all_masks = [], []
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            masks = batch["mask"]
            logits = model(images)
            preds = logits.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_masks.append(masks)

    all_preds = torch.cat(all_preds)
    all_masks = torch.cat(all_masks)
    results = compute_metrics(all_preds, all_masks, num_classes=mcfg["num_classes"])

    print("=" * 40)
    print(f"Test results on: {test_dir}")
    print(f"  IoU:       {results['iou']:.4f}")
    print(f"  Dice:      {results['dice']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
