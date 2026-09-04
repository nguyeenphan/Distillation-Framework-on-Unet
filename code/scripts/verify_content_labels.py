"""
verify_content_labels.py — Is y still valid for x_content?
──────────────────────────────────────────────────────────
The content twin mixes U,V from a DIFFERENT leaf, so the structure in
x_content is no longer purely leaf i's — but L_cont still trains it
against y_i.  So the mix strength is a correctness constraint, not a
tuning knob.

Measured on this dataset: top_k_ranks saturates past ~16 (SVD energy sits
in the leading ranks), so content_t is the knob that actually moves the
image.  Sweeps t with the mask overlaid; pick the LARGEST t where the
lesion boundary still sits inside the red overlay.

    python scripts/verify_content_labels.py --config configs/spectral.yaml
    python scripts/verify_content_labels.py --ks 3,8,16,32 --n 4
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
from scripts.visualize_stycona import _chw_to_hwc_uint8, _overlay_mask, _add_label


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/spectral.yaml")
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--ks", default="16", help="top_k_ranks values to sweep")
    p.add_argument("--ts", default="0.15,0.3,0.5,0.7",
                   help="content_t values to sweep — this is the knob that matters")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--out", default="viz_content")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    dcfg, sc = cfg["data"], cfg["stycona"]
    ks = [int(v) for v in args.ks.split(",")]
    ts = [float(v) for v in args.ts.split(",")]

    ds = LeafDiseaseDataset(
        root_dir=dcfg[f"{args.split}_dir"],
        image_size=dcfg["image_size"],
        transform=None,
        base_image_only=dcfg.get("base_image_only", False),
        auxiliary_from_styles=dcfg.get("auxiliary_from_styles", False),
    )

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    idxs = np.linspace(0, len(ds) - 1, min(args.n, len(ds))).astype(int)

    for n_i, idx in enumerate(idxs):
        rec = ds[int(idx)]
        orig = _chw_to_hwc_uint8(rec["image"])
        mask = rec["mask"].numpy()
        # Content donor: a different leaf, matching what the trainer feeds.
        donor = ds._sample_other_leaf(rec["content_id"])

        cells = [_add_label(_overlay_mask(orig, mask), "original + y"),
                 _add_label(donor, "content donor (other leaf)")]
        for k in ks:
            for t in ts:
                aug = StyConaAugmentor(
                    style_alpha_range=tuple(sc["style_alpha_range"]),
                    content_mix_enabled=True,
                    content_t=t,
                    top_k_ranks=k,
                    per_channel=sc["per_channel_svd"],
                )
                x_content = aug(orig, donor, mode="content").astype("uint8")
                cells.append(_add_label(_overlay_mask(x_content, mask),
                                        f"t={t} k={k} + y"))

        panel = np.concatenate(cells, axis=1)
        fname = out_dir / f"{args.split}_{rec['content_id']}_{n_i:02d}.png"
        cv2.imwrite(str(fname), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        print(f"  saved {fname}")

    print(f"\nDone → {out_dir.resolve()}")
    print("Chọn t LỚN NHẤT mà biên tổn thương còn nằm trong vùng overlay đỏ,")
    print("rồi đặt vào stycona.content_mix.t của config.")
    print("(k đã đo là bão hoà sau ~16 — t mới là nút vặn thật.)")


if __name__ == "__main__":
    main()
