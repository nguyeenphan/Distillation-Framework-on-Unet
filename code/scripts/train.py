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

from augmentation.stycona import StyConaAugmentor
from augmentation.view_generator import ViewGenerator
from data.dataset import LeafDiseaseDataset
from data.transforms import get_train_transform, get_val_transform
from models.resnet18_unet import ResNet18UNet
from trainers.mean_teacher import MeanTeacherTrainer
from trainers.spectral import SpectralTrainer
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
    sc = cfg["stycona"]
    stycona = StyConaAugmentor(
        style_alpha_range=tuple(sc["style_alpha_range"]),
        content_mix_enabled=sc["content_mix"]["enabled"],
        content_t=sc["content_mix"]["t"],
        top_k_ranks=sc["content_mix"]["top_k_ranks"],
        per_channel=sc["per_channel_svd"],
    ) if sc["enabled"] else None
    view_gen = ViewGenerator(
        strong_cfg=cfg["views"]["strong"],
        weak_cfg=cfg["views"]["weak"],
    )

    train_ds = LeafDiseaseDataset(
        root_dir=dcfg["train_dir"],
        image_size=dcfg["image_size"],
        transform=get_train_transform(dcfg["image_size"]),
        base_image_only=dcfg.get("base_image_only", False),
        auxiliary_from_styles=dcfg.get("auxiliary_from_styles", False),
        stycona=stycona,
        view_gen=view_gen,
        spectral=dcfg.get("spectral", False),
        style_twin=dcfg.get("style_twin", "spectral"),
    )
    # Validation: evaluate on the real photos only (no CAST styles, no StyCona).
    val_ds = LeafDiseaseDataset(
        root_dir=dcfg["val_dir"],
        image_size=dcfg["image_size"],
        transform=get_val_transform(dcfg["image_size"]),
        base_image_only=dcfg.get("base_image_only", False),
    )

    pin_memory = device.type == "cuda"
    num_workers = dcfg["num_workers"]
    # 'spawn' starts clean worker processes — required on macOS because 'fork'
    # copies BLAS thread state from the parent, which causes numpy SVD to segfault.
    mp_context = "spawn" if num_workers > 0 else None
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        multiprocessing_context=mp_context,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        multiprocessing_context=mp_context,
        persistent_workers=num_workers > 0,
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
    if dcfg.get("spectral", False):
        trainer = SpectralTrainer(cfg, student, device)
    else:
        trainer = MeanTeacherTrainer(cfg, student, device)
    trainer.train(train_loader, val_loader)

    print("Training complete.")


if __name__ == "__main__":
    main()
