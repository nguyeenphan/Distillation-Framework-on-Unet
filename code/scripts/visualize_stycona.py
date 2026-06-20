"""
visualize_stycona.py — See StyCona augmentation on your real data
─────────────────────────────────────────────────────────────────
Loads a few samples from a split, runs each through StyConaAugmentor
(original + same-leaf auxiliary), and saves a side-by-side panel:

    [ original | auxiliary (xj) | StyCona augmented | mask overlay ]

Usage:
    python scripts/visualize_stycona.py --config configs/default.yaml
    python scripts/visualize_stycona.py --split train --n 6 --out viz
"""

import argparse
import sys
from pathlib import Path

import yaml
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import LeafDiseaseDataset
from augmentation.stycona import StyConaAugmentor


def _chw_to_hwc_uint8(t):
    return (t.permute(1, 2, 0).numpy() * 255).astype("uint8")


def _overlay_mask(img_uint8, mask, color=(255, 0, 0), alpha=0.45):
    out = img_uint8.copy()
    m = mask.astype(bool)
    out[m] = ((1 - alpha) * out[m] + alpha * np.array(color)).astype("uint8")
    return out


def _add_label(img_uint8, text, strip_h=26):
    """Append a black caption strip with centered white text below an RGB image."""
    h, w = img_uint8.shape[:2]
    strip = np.zeros((strip_h, w, 3), dtype="uint8")
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x = max((w - tw) // 2, 2)
    y = (strip_h + th) // 2
    cv2.putText(strip, text, (x, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return np.concatenate([img_uint8, strip], axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--n", type=int, default=6, help="number of samples to render")
    p.add_argument("--out", default="viz", help="output directory")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    dcfg, sc = cfg["data"], cfg["stycona"]
    image_size = dcfg["image_size"]

    ds = LeafDiseaseDataset(
        root_dir=dcfg[f"{args.split}_dir"],
        image_size=image_size,
        transform=None,  # no spatial aug → clean visualization
        base_image_only=dcfg.get("base_image_only", False),
        auxiliary_from_styles=dcfg.get("auxiliary_from_styles", False),
    )

    aug = StyConaAugmentor(
        style_alpha_range=tuple(sc["style_alpha_range"]),
        content_mix_enabled=sc["content_mix"]["enabled"],
        content_t=sc["content_mix"]["t"],
        top_k_ranks=sc["content_mix"]["top_k_ranks"],
        per_channel=sc["per_channel_svd"],
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = min(args.n, len(ds))
    idxs = np.linspace(0, len(ds) - 1, n).astype(int)

    for k, idx in enumerate(idxs):
        rec = ds[int(idx)]
        orig = _chw_to_hwc_uint8(rec["image"])
        auxi = _chw_to_hwc_uint8(rec["auxiliary"])
        mask = rec["mask"].numpy()

        aug_np = aug(orig, auxi).astype("uint8")
        overlay = _overlay_mask(aug_np, mask)

        cells = [
            _add_label(orig, "original"),
            _add_label(auxi, "auxiliary (xj)"),
            _add_label(aug_np, "StyCona"),
            _add_label(overlay, "mask overlay"),
        ]
        panel = np.concatenate(cells, axis=1)
        panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
        fname = out_dir / f"{args.split}_{rec['content_id']}_{k:02d}.png"
        cv2.imwrite(str(fname), panel_bgr)
        print(f"  saved {fname}  (original | auxiliary | stycona | mask-overlay)")

    print(f"\nDone. {n} panels written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
