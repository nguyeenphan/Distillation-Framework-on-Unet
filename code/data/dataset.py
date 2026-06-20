"""
LeafDiseaseDataset
──────────────────
Flat folder layout — images and masks live side by side:

    dataset/train/
        00001_img.png             # base image
        00001_seg.png             # base mask
        00001_style0_img.png      # CAST-restyled image
        00001_style0_seg.png      # corresponding mask
        00001_style1_img.png
        00001_style1_seg.png
        ...

Naming convention:
    image :  {content_id}[_style{n}]_img.png
    mask  :  {content_id}[_style{n}]_seg.png

The base image has no _style{n} segment; restyled variants do.
The dataset groups images by content_id so that StyCona can sample
a random CAST auxiliary from the *same* leaf (different style).
"""

import os
import re
import random
from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# Disease class is stored as gray value ~38 in the _seg masks (background = 0).
# Anything above this midpoint counts as disease.
_DISEASE_THRESHOLD = 19

# Parse an image stem like  00001_style3_img  or  00001_img
#   → content_id="00001", style_idx="3" (or "base" when no _style segment)
_PATTERN = re.compile(r"^(?P<cid>.+?)(?:_style(?P<sty>\d+))?_img$")


class LeafDiseaseDataset(Dataset):

    def __init__(
        self,
        root_dir: str,
        image_size: int = 256,
        transform: Optional[Callable] = None,
        base_image_only: bool = False,
        auxiliary_from_styles: bool = False,
    ):
        """
        Args:
            base_image_only:       if True, only the base photo `{id}_img.png`
                                   becomes a supervised training sample (the CAST
                                   `_style{n}` variants are kept only as auxiliaries).
            auxiliary_from_styles: if True, StyCona's auxiliary is drawn only from
                                   the CAST `_style{n}` variants of the same leaf
                                   (never the base photo).
        """
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.transform = transform
        self.auxiliary_from_styles = auxiliary_from_styles

        # ── Discover all image / mask pairs ──────────────
        # records holds EVERY image (base + styles); samples is the subset used
        # as supervised training items; content_pool indexes records per leaf.
        self.records: list[dict] = []
        self.content_pool: dict[str, list[int]] = {}   # content_id → [index into records]

        all_stems = {f.stem for f in self.root_dir.glob("*.png")}

        for stem in sorted(all_stems):
            if not stem.endswith("_img"):
                continue  # only iterate over images; masks are paired below

            # Corresponding mask:  ..._img  →  ..._seg
            mask_stem = stem[: -len("_img")] + "_seg"
            if mask_stem not in all_stems:
                continue  # no mask → skip

            # Parse content_id / style
            m = _PATTERN.match(stem)
            if m:
                content_id = m.group("cid")               # e.g. "00001"
                style_idx = m.group("sty") or "base"       # e.g. "3" or "base"
            else:
                content_id = stem                          # fallback
                style_idx = "base"

            rec_idx = len(self.records)
            self.records.append({
                "image": self.root_dir / f"{stem}.png",
                "mask":  self.root_dir / f"{mask_stem}.png",
                "content_id": content_id,
                "style_idx": style_idx,
            })
            self.content_pool.setdefault(content_id, []).append(rec_idx)

        # Which records are supervised training samples?
        if base_image_only:
            self.samples = [i for i, r in enumerate(self.records)
                            if r["style_idx"] == "base"]
        else:
            self.samples = list(range(len(self.records)))

        assert len(self.samples) > 0, f"No image/mask pairs found in {root_dir}"
        print(f"  {root_dir}: {len(self.samples)} samples "
              f"({len(self.records)} images, {len(self.content_pool)} unique leaves)"
              f"{' [base-only]' if base_image_only else ''}")

    # ──────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        rec_idx = self.samples[idx]
        rec = self.records[rec_idx]

        # ── Load image & mask ────────────────
        image = self._read_image(rec["image"])
        mask = self._read_mask(rec["mask"])

        # ── Pick a CAST auxiliary (same leaf, different style) ──
        auxiliary = self._sample_auxiliary(rec["content_id"], rec_idx)

        # ── Spatial transforms (image + mask together) ──
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        auxiliary = cv2.resize(auxiliary, (self.image_size, self.image_size))

        # ── To tensors ───────────────────────
        image = self._to_tensor(image)
        auxiliary = self._to_tensor(auxiliary)
        mask = torch.from_numpy(mask).long()

        return {
            "image": image,            # [3, H, W]
            "auxiliary": auxiliary,     # [3, H, W]
            "mask": mask,              # [H, W]
            "content_id": rec["content_id"],
        }

    # ── helpers ──────────────────────────────

    def _sample_auxiliary(self, content_id: str, current_idx: int) -> np.ndarray:
        """Pick a random CAST variant of the same leaf (different from current)."""
        pool = self.content_pool.get(content_id, [current_idx])
        candidates = [i for i in pool if i != current_idx]
        if self.auxiliary_from_styles:
            # Restrict donors to the CAST `_style{n}` variants (exclude base photo).
            styles = [i for i in candidates
                      if self.records[i]["style_idx"] != "base"]
            if styles:
                candidates = styles
        if not candidates:
            candidates = [current_idx]  # leaf has no other variant → use itself
        aux_idx = random.choice(candidates)
        return self._read_image(self.records[aux_idx]["image"])

    def _read_image(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        return img

    def _read_mask(self, path: Path) -> np.ndarray:
        m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        m = cv2.resize(m, (self.image_size, self.image_size),
                       interpolation=cv2.INTER_NEAREST)
        # Disease pixels are encoded as a dark-gray value (~38), background = 0.
        # Threshold at the midpoint to keep lesion cores and drop anti-aliased
        # boundary pixels.  (NOT > 127 — these masks never reach 255.)
        m = (m > _DISEASE_THRESHOLD).astype(np.uint8)
        return m

    @staticmethod
    def _to_tensor(img: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0)
