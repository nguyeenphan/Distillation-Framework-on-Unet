"""
train.py — Entry point
──────────────────────
Usage:
    python scripts/train.py --config configs/default.yaml
"""

import argparse
import sys
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import LeafDiseaseDataset
from data.transforms import get_train_transform, get_val_transform
from models.resnet18_unet import ResNet18UNet
from trainers.mean_teacher import MeanTeacherTrainer
from utils.helpers import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    # ── Load config ──────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Datasets ─────────────────────────────
    dcfg = cfg["data"]

    train_ds = LeafDiseaseDataset(
        root_dir=dcfg["train_dir"],
        image_size=dcfg["image_size"],
        transform=get_train_transform(dcfg["image_size"]),
        base_image_only=dcfg.get("base_image_only", False),
        auxiliary_from_styles=dcfg.get("auxiliary_from_styles", False),
    )
    # Validation: evaluate on the real photos only (no CAST styles, no StyCona).
    val_ds = LeafDiseaseDataset(
        root_dir=dcfg["val_dir"],
        image_size=dcfg["image_size"],
        transform=get_val_transform(dcfg["image_size"]),
        base_image_only=dcfg.get("base_image_only", False),
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=dcfg["num_workers"],
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=dcfg["num_workers"],
        pin_memory=pin_memory,
    )

    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    # ── Model ────────────────────────────────
    mcfg = cfg["model"]
    student = ResNet18UNet(
        num_classes=mcfg["num_classes"],
        pretrained=mcfg["pretrained"],
        in_channels=mcfg["in_channels"],
        use_aspp=mcfg.get("use_aspp", True),
        aspp_rates=tuple(mcfg.get("aspp_rates", [6, 12, 18])),
    )
    print(f"Model params: {sum(p.numel() for p in student.parameters()):,}")

    # ── Train ────────────────────────────────
    trainer = MeanTeacherTrainer(cfg, student, device)
    trainer.train(train_loader, val_loader)

    print("Training complete.")


if __name__ == "__main__":
    main()
