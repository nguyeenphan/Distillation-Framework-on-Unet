"""
evaluate.py — Run inference on test set or unseen domains
─────────────────────────────────────────────────────────
Usage:
    python scripts/evaluate.py --config configs/default.yaml \
        --checkpoint checkpoints/best.pth

    # Save visual comparisons:
    python scripts/evaluate.py --config configs/default.yaml \
        --checkpoint checkpoints/best.pth --vis_dir ./visualizations

    # Custom test directory:
    python scripts/evaluate.py --config configs/default.yaml \
        --checkpoint checkpoints/best.pth --test_dir ./unseen_domain
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import LeafDiseaseDataset
from data.transforms import get_val_transform
from models.resnet18_unet import ResNet18UNet
from utils.helpers import load_checkpoint
from utils.metrics import compute_metrics


# Disease overlay colour (BGR): semi-transparent red
_OVERLAY_COLOR = (0, 0, 220)
_OVERLAY_ALPHA = 0.45


def _tensor_to_bgr(t: torch.Tensor) -> np.ndarray:
    """(C, H, W) float [0,1] → (H, W, 3) uint8 BGR."""
    rgb = (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _apply_overlay(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend a coloured overlay onto disease pixels."""
    out = bgr.copy().astype(np.float32)
    disease = mask.astype(bool)
    out[disease] = (
        (1 - _OVERLAY_ALPHA) * out[disease]
        + _OVERLAY_ALPHA * np.array(_OVERLAY_COLOR, dtype=np.float32)
    )
    return out.clip(0, 255).astype(np.uint8)


def _save_comparison(
    image_t: torch.Tensor,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    out_path: Path,
    label: str,
) -> None:
    """Save a 3-panel side-by-side PNG: original | ground truth | prediction."""
    h, w = gt_mask.shape
    bgr = _tensor_to_bgr(image_t)

    gt_vis   = _apply_overlay(bgr, gt_mask)
    pred_vis = _apply_overlay(bgr, pred_mask)

    # Panel headers (black bar above each panel)
    bar_h = 28
    font  = cv2.FONT_HERSHEY_SIMPLEX
    bar_color = (30, 30, 30)

    def _add_header(img, text):
        bar = np.full((bar_h, w, 3), bar_color, dtype=np.uint8)
        cv2.putText(bar, text, (6, 20), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return np.vstack([bar, img])

    col0 = _add_header(bgr,      "Original")
    col1 = _add_header(gt_vis,   "Ground Truth")
    col2 = _add_header(pred_vis, "Prediction")

    grid = np.hstack([col0, col1, col2])

    # Thin title bar at the very top
    title_bar = np.full((30, grid.shape[1], 3), (15, 15, 15), dtype=np.uint8)
    cv2.putText(title_bar, label, (8, 21), font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    grid = np.vstack([title_bar, grid])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_dir",   type=str, default=None)
    parser.add_argument("--vis_dir",    type=str, default=None,
                        help="Save visual comparisons to this folder (optional)")
    parser.add_argument("--base_only", action="store_true",
                        help="Evaluate base images only (default: all including styles)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    test_dir = args.test_dir or cfg["data"]["test_dir"]
    vis_dir  = Path(args.vis_dir) if args.vis_dir else None

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
    image_size = cfg["data"]["image_size"]
    test_ds = LeafDiseaseDataset(
        root_dir=test_dir,
        image_size=image_size,
        transform=get_val_transform(image_size),
        base_image_only=args.base_only,
    )
    test_loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"], shuffle=False)

    # ── Inference ────────────────────────────
    all_preds, all_masks = [], []
    sample_idx = 0

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            masks  = batch["mask"]
            logits = model(images)
            preds  = logits.argmax(dim=1).cpu()

            all_preds.append(preds)
            all_masks.append(masks)

            # ── Save visualizations ──────────
            if vis_dir is not None:
                for i in range(images.shape[0]):
                    rec = test_ds.records[test_ds.samples[sample_idx]]
                    stem = rec["image"].stem          # e.g. "00001_img"
                    label = f"{stem}   Dice={_dice(preds[i], masks[i]):.3f}"
                    _save_comparison(
                        image_t   = batch["image"][i],
                        gt_mask   = masks[i].numpy(),
                        pred_mask = preds[i].numpy(),
                        out_path  = vis_dir / f"{stem}.png",
                        label     = label,
                    )
                    sample_idx += 1

    all_preds = torch.cat(all_preds)
    all_masks = torch.cat(all_masks)
    results   = compute_metrics(all_preds, all_masks, num_classes=mcfg["num_classes"])

    print("=" * 40)
    print(f"Test results on: {test_dir}")
    print(f"  IoU:       {results['iou']:.4f}")
    print(f"  Dice:      {results['dice']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print("=" * 40)
    if vis_dir:
        n = len(test_ds)
        print(f"  Saved {n} comparison images → {vis_dir}/")


def _dice(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Quick per-image Dice for the title label."""
    p = (pred == 1)
    g = (gt   == 1)
    inter = (p & g).sum().item()
    denom = p.sum().item() + g.sum().item()
    return (2 * inter / denom) if denom > 0 else 1.0


if __name__ == "__main__":
    main()
