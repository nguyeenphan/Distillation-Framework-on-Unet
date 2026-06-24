"""
viz/stycona_cases.py -Full StyCona case breakdown
────────────────────────────────────────────────────
Renders a 4-row × 3-col grid for each sample:

  Row 1 -Inputs
    [ Original (xi) ]  [ GT mask overlay ]  [ Auxiliary (xj / CAST) ]

  Row 2 -Style blend only  (content_mix OFF, α varies)
    [ α=0.30 ]  [ α=0.50 ]  [ α=0.70 ]

  Row 3 -Full StyCona  (style blend + content mix, α varies)
    [ α=0.30 ]  [ α=0.50 ]  [ α=0.70 ]

  Row 4 -Views fed to Student / Teacher  (default α from config)
    [ StyCona output ]  [ Strong view (Student) ]  [ Weak view (Teacher) ]

Usage:
    python viz/stycona_cases.py --config configs/default.yaml
    python viz/stycona_cases.py --split train --n 4 --out viz/cases
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from augmentation.stycona import StyConaAugmentor
from augmentation.view_generator import ViewGenerator
from data.dataset import LeafDiseaseDataset


def _load_image_np(path: Path, size: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    return img


# ── visual constants ─────────────────────────────
DISEASE_COLOR  = np.array([220,  50,  50], dtype=np.float32)   # red (RGB)
OVERLAY_ALPHA  = 0.50
HEADER_H       = 28
TITLE_H        = 36
SECTION_H      = 30
DIVIDER_H      = 4
FONT           = cv2.FONT_HERSHEY_SIMPLEX
BG_DARK        = (20,  20,  20)
BG_SECTION     = (45,  45,  45)
BG_TITLE       = (10,  10,  10)
TEXT_COLOR     = (240, 240, 240)
DIVIDER_COLOR  = (80,  80, 200)    # blue divider between sections


# ── image helpers ────────────────────────────────

def _to_uint8(arr) -> np.ndarray:
    """Convert tensor (C,H,W) or numpy (H,W,C) float [0-255] → uint8 RGB."""
    if isinstance(arr, torch.Tensor):
        arr = arr.permute(1, 2, 0).numpy()
    return np.clip(arr, 0, 255).astype(np.uint8)


def _overlay(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = img.astype(np.float32).copy()
    m = mask.astype(bool)
    out[m] = (1 - OVERLAY_ALPHA) * out[m] + OVERLAY_ALPHA * DISEASE_COLOR
    return out.clip(0, 255).astype(np.uint8)


def _label(img: np.ndarray, text: str, bg=BG_DARK) -> np.ndarray:
    """Add a caption strip BELOW the image."""
    w = img.shape[1]
    strip = np.full((HEADER_H, w, 3), bg, dtype=np.uint8)
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.48, 1)
    x = max((w - tw) // 2, 4)
    y = (HEADER_H + th) // 2
    cv2.putText(strip, text, (x, y), FONT, 0.48, TEXT_COLOR, 1, cv2.LINE_AA)
    return np.vstack([img, strip])


def _section_bar(width: int, text: str) -> np.ndarray:
    bar = np.full((SECTION_H, width, 3), BG_SECTION, dtype=np.uint8)
    (tw, _), _ = cv2.getTextSize(text, FONT, 0.55, 1)
    cv2.putText(bar, text, (8, 21), FONT, 0.55, (180, 220, 255), 1, cv2.LINE_AA)
    return bar


def _divider(width: int) -> np.ndarray:
    d = np.full((DIVIDER_H, width, 3), DIVIDER_COLOR, dtype=np.uint8)
    return d


def _title_bar(width: int, text: str) -> np.ndarray:
    bar = np.full((TITLE_H, width, 3), BG_TITLE, dtype=np.uint8)
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.65, 1)
    x = max((width - tw) // 2, 8)
    y = (TITLE_H + th) // 2
    cv2.putText(bar, text, (x, y), FONT, 0.65, (200, 200, 200), 1, cv2.LINE_AA)
    return bar


def _hstack(cells) -> np.ndarray:
    return np.hstack(cells)


def _stycona_fixed_alpha(aug_base: StyConaAugmentor, img: np.ndarray,
                         aux: np.ndarray, alpha: float,
                         content_mix: bool) -> np.ndarray:
    """Run StyCona with a deterministic alpha (bypass random sampling)."""
    tmp = StyConaAugmentor(
        style_alpha_range=(alpha, alpha),
        content_mix_enabled=content_mix,
        content_t=aug_base.content_t,
        top_k_ranks=aug_base.top_k_ranks,
        per_channel=aug_base.per_channel,
    )
    return _to_uint8(tmp(img, aux))


# ── main grid builder ────────────────────────────

def _pad_row_to_width(row: np.ndarray, target_w: int, bg=BG_DARK) -> np.ndarray:
    """Right-pad a row with a solid color block if narrower than target_w."""
    h, w = row.shape[:2]
    if w >= target_w:
        return row
    pad = np.full((h, target_w - w, 3), bg, dtype=np.uint8)
    return np.hstack([row, pad])


def build_grid(
    img_uint8: np.ndarray,
    aux_uint8: np.ndarray,
    mask: np.ndarray,
    aug: StyConaAugmentor,
    view_gen: ViewGenerator,
    content_id: str,
    all_styles: list[np.ndarray] | None = None,   # list of CAST style images (RGB uint8)
) -> np.ndarray:

    W = img_uint8.shape[1]

    # ── Row 1: Inputs ────────────────────────────
    gt_vis = _overlay(img_uint8, mask)
    r1 = _hstack([
        _label(img_uint8, "Original  (xi)"),
        _label(gt_vis,    "GT mask overlay"),
        _label(aux_uint8, "Auxiliary  (xj / CAST)"),
    ])

    # ── Row 2: Style blend only ──────────────────
    style_cells = []
    for alpha in (0.30, 0.50, 0.70):
        out = _stycona_fixed_alpha(aug, img_uint8, aux_uint8, alpha, content_mix=False)
        style_cells.append(_label(out, f"Style only  alpha={alpha:.2f}"))
    r2 = _hstack(style_cells)

    # ── Row 3: Full StyCona ──────────────────────
    full_cells = []
    for alpha in (0.30, 0.50, 0.70):
        out = _stycona_fixed_alpha(aug, img_uint8, aux_uint8, alpha, content_mix=True)
        full_cells.append(_label(out, f"Style+Content  alpha={alpha:.2f}"))
    r3 = _hstack(full_cells)

    # ── Row 4: Views ────────────────────────────
    aug_default = _to_uint8(aug(img_uint8, aux_uint8))
    aug_t = torch.from_numpy(aug_default.transpose(2, 0, 1)).float().div(255.0)
    strong_t, weak_t = view_gen(aug_t)

    strong_np = (strong_t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    weak_np   = (weak_t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

    r4 = _hstack([
        _label(aug_default, "StyCona output  (random alpha)"),
        _label(strong_np,   "Strong view  → Student"),
        _label(weak_np,     "Weak view    → Teacher"),
    ])

    # ── Row 5 & 6: All CAST styles + StyCona with each ──
    r5 = r6 = None
    if all_styles:
        style_cells, stycona_cells = [], []
        for i, sty in enumerate(all_styles):
            style_cells.append(_label(sty, f"style{i}"))
            out = _stycona_fixed_alpha(aug, img_uint8, sty, 0.50, content_mix=True)
            stycona_cells.append(_label(out, f"StyCona  xi + style{i}"))
        r5 = _hstack(style_cells)
        r6 = _hstack(stycona_cells)

    # ── Assemble with section bars + dividers ────
    # All rows must share the same width -use the widest one
    candidate_widths = [r1.shape[1], r2.shape[1], r3.shape[1], r4.shape[1]]
    if r5 is not None:
        candidate_widths += [r5.shape[1], r6.shape[1]]
    grid_w = max(candidate_widths)

    def _pad(row):
        return _pad_row_to_width(row, grid_w)

    rows = [
        _title_bar(grid_w, f"StyCona cases  - leaf: {content_id}"),
        _section_bar(grid_w, "ROW 1 | Inputs"),
        _pad(r1),
        _divider(grid_w),
        _section_bar(grid_w, "ROW 2 | Style blend only  (singular values sigma)"),
        _pad(r2),
        _divider(grid_w),
        _section_bar(grid_w, "ROW 3 | Full StyCona  (sigma blend + top-k U,V content mix)"),
        _pad(r3),
        _divider(grid_w),
        _section_bar(grid_w, "ROW 4 | Views generated from StyCona output"),
        _pad(r4),
    ]
    if r5 is not None:
        rows += [
            _divider(grid_w),
            _section_bar(grid_w, "ROW 5 | All CAST style variants for this leaf"),
            _pad(r5),
            _divider(grid_w),
            _section_bar(grid_w, "ROW 6 | StyCona output  xi + each style  (alpha=0.50)"),
            _pad(r6),
        ]
    return np.vstack(rows)


# ── CLI ──────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--split",  default="train", choices=["train", "val", "test"])
    p.add_argument("--n",      type=int, default=4, help="number of leaves to render")
    p.add_argument("--out",    default="viz/cases", help="output directory")
    args = p.parse_args()

    cfg  = yaml.safe_load(open(args.config))
    dcfg = cfg["data"]
    sc   = cfg["stycona"]
    vc   = cfg["views"]

    ds = LeafDiseaseDataset(
        root_dir=dcfg[f"{args.split}_dir"],
        image_size=dcfg["image_size"],
        transform=None,
        base_image_only=True,           # one base image per leaf for clarity
        auxiliary_from_styles=True,
    )

    aug = StyConaAugmentor(
        style_alpha_range=tuple(sc["style_alpha_range"]),
        content_mix_enabled=sc["content_mix"]["enabled"],
        content_t=sc["content_mix"]["t"],
        top_k_ranks=sc["content_mix"]["top_k_ranks"],
        per_channel=sc["per_channel_svd"],
    )
    view_gen = ViewGenerator(strong_cfg=vc["strong"], weak_cfg=vc["weak"])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n    = min(args.n, len(ds))
    idxs = np.linspace(0, len(ds) - 1, n).astype(int)

    image_size = dcfg["image_size"]

    for k, idx in enumerate(idxs):
        rec      = ds[int(idx)]
        img_np   = (rec["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        aux_np   = (rec["auxiliary"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        mask_np  = rec["mask"].numpy()
        cid      = rec["content_id"]

        # Collect all CAST style variants for this leaf from disk
        rec_meta   = ds.records[ds.samples[int(idx)]]
        root       = rec_meta["image"].parent
        all_styles = []
        for sty_idx in range(20):   # scan style0 … style19, stop when absent
            p = root / f"{cid}_style{sty_idx}_img.png"
            if not p.exists():
                break
            all_styles.append(_load_image_np(p, image_size))

        grid_rgb = build_grid(img_np, aux_np, mask_np, aug, view_gen, cid,
                              all_styles=all_styles or None)
        grid_bgr = cv2.cvtColor(grid_rgb, cv2.COLOR_RGB2BGR)

        out_path = out_dir / f"{args.split}_{cid}_{k:02d}.png"
        cv2.imwrite(str(out_path), grid_bgr)
        print(f"  saved {out_path}  ({len(all_styles)} style variants found)")

    print(f"\nDone. {n} grids written to {out_dir.resolve()}/")


if __name__ == "__main__":
    main()
