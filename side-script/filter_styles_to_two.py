from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path


# --- CONFIG: chỉnh đường dẫn và tùy chọn tại đây ---
INPUT_DIR = Path("/Users/nguyenphan/Developer/Distillation-Framework-on-Unet/dataset/val")
OUTPUT_DIR = Path("/Users/nguyenphan/Developer/Distillation-Framework-on-Unet/copy_dataset/val")

# Copy kèm {id}_img / {id}_seg nếu có
COPY_BASE_IMAGES = True

# True = chỉ in thao tác, không ghi file
DRY_RUN = False

# style nguồn -> style đích sau đổi tên (vd: 3->0, 6->1)
STYLE_MAP = {3: 0, 6: 1}
# --- hết CONFIG ---


# {id}_style{N}_{img|seg}.ext  hoặc  {id}_{img|seg}.ext
FILENAME_RE = re.compile(
    r"^(\d+)_(?:(?:style(\d+))_)?(img|seg)\.(png|jpg|jpeg|webp)$",
    re.IGNORECASE,
)


def parse_name(path: Path) -> tuple[str, int | None, str, str] | None:
    """
    Trả về (id_str, style_index hoặc None cho ảnh gốc, kind img|seg, ext) hoặc None.
    """
    m = FILENAME_RE.match(path.name)
    if not m:
        return None
    id_str, style_s, kind, ext = m.groups()
    style = int(style_s) if style_s is not None else None
    return id_str, style, kind, ext.lower()


def collect_ids(input_dir: Path) -> set[str]:
    ids: set[str] = set()
    for p in input_dir.iterdir():
        if not p.is_file():
            continue
        parsed = parse_name(p)
        if parsed:
            ids.add(parsed[0])
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Mặc định dùng INPUT_DIR / OUTPUT_DIR trong file. "
            "Có thể ghi đè bằng --input / --output."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Ghi đè INPUT_DIR trong CONFIG.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Ghi đè OUTPUT_DIR trong CONFIG.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ in ra thao tác, không copy file (ghi đè DRY_RUN trong CONFIG).",
    )
    parser.add_argument(
        "--no-base",
        action="store_true",
        help="Không copy {id}_img / {id}_seg (ghi đè COPY_BASE_IMAGES=False).",
    )
    args = parser.parse_args()

    input_dir = (args.input or INPUT_DIR).resolve()
    output_dir = (args.output or OUTPUT_DIR).resolve()
    dry_run = DRY_RUN or args.dry_run
    copy_base = COPY_BASE_IMAGES and not args.no_base

    if not input_dir.is_dir():
        print(f"Lỗi: không tìm thấy thư mục đầu vào: {input_dir}", file=sys.stderr)
        return 1

    style_keep = dict(STYLE_MAP)
    ids = sorted(collect_ids(input_dir), key=lambda x: int(x))

    if not ids:
        print(f"Cảnh báo: không parse được file nào trong {input_dir}", file=sys.stderr)

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    errors: list[str] = []

    for id_str in ids:
        # Ảnh gốc không có style
        if copy_base:
            for kind in ("img", "seg"):
                src = None
                for ext in ("png", "jpg", "jpeg", "webp"):
                    cand = input_dir / f"{id_str}_{kind}.{ext}"
                    if cand.is_file():
                        src = cand
                        break
                if src is None:
                    continue
                dst = output_dir / src.name
                if dry_run:
                    print(f"[dry-run] copy {src} -> {dst}")
                else:
                    shutil.copy2(src, dst)
                copied += 1

        # style3 -> style0, style6 -> style1
        for old_style, new_style in style_keep.items():
            for kind in ("img", "seg"):
                src = None
                for ext in ("png", "jpg", "jpeg", "webp"):
                    cand = input_dir / f"{id_str}_style{old_style}_{kind}.{ext}"
                    if cand.is_file():
                        src = cand
                        break
                if src is None:
                    errors.append(
                        f"Thiếu: {id_str}_style{old_style}_{kind}.* trong {input_dir}"
                    )
                    skipped += 1
                    continue
                ext = src.suffix.lstrip(".").lower()
                dst_name = f"{id_str}_style{new_style}_{kind}.{ext}"
                dst = output_dir / dst_name
                if dry_run:
                    print(f"[dry-run] copy {src.name} -> {dst_name}")
                else:
                    shutil.copy2(src, dst)
                copied += 1

    for msg in errors:
        print(msg, file=sys.stderr)

    print(
        f"Xong: {copied} file đã {'(dry-run) ' if dry_run else ''}chuẩn bị/ghi, "
        f"{len(ids)} id."
    )
    if skipped:
        print(f"Cảnh báo: {skipped} file mong đợi nhưng không tìm thấy.", file=sys.stderr)

    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
