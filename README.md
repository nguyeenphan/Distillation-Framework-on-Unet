# Distillation-Framework-on-Unet

Project đã được setup skeleton huấn luyện StyCona trong `code/stylecona` để chạy trực tiếp với dataset local của bạn.

## 1) Cấu trúc dataset đang hỗ trợ

Pipeline đọc dữ liệu theo cặp file:

- `*_img.png`: ảnh input
- `*_seg.png`: mask segmentation

**Ảnh gốc vs ảnh style (StyCona):**

- Ảnh **gốc** để học: `{id}_img.png` + `{id}_seg.png` — không chứa chuỗi `_style` trong tên file.
- Ảnh **auxiliary / style** (cùng id): `{id}_style0_img.png`, `{id}_style1_img.png`, … — chỉ dùng làm nhánh style trong `StyConTransform`; mask `{id}_styleK_seg.png` không dùng trong loss hiện tại.
- Trong `config/default.yaml`, `stycona.auxiliary_source: paired_styles` khiến mỗi step lấy đúng file style **cùng id** (chọn biến thể K theo `style_variant_sampling`), **không** random ảnh khác trong batch.

### Layout phẳng (khuyến nghị): `train` / `val` / `test` ngay dưới `dataset/`

```text
dataset/
  train/
  val/
  test/
```

Trong `config/default.yaml` đặt `data.layout: flat` và `train_split`, `val_split`, `test_split` (test chỉ để thống nhất tên folder; script train hiện chỉ dùng train + val). Metric `target` dùng cùng thư mục với `val` trừ khi bạn set `target_val_split` trỏ sang folder val đích khác.

### Layout lồng (domain): như cũ `original/` và `augmented/`

```text
dataset/
  original/
    train/ val/
  augmented/
    val/
```

Đặt `data.layout: nested`, `source_domain`, `target_domain`, và có thể chỉnh `train_split` / `val_split` (mặc định `train`, `val`).

## 2) Cài đặt môi trường

```bash
cd code/stylecona
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3) Chạy train StyCona

Mặc định:

```bash
cd code/stylecona
python3 train.py --config config/default.yaml
```

Nếu cần debug trực quan StyCona mỗi epoch:

```bash
cd code/stylecona
python3 train.py --config config/default.yaml --stycona-debug-save --stycona-debug-max-samples 4
```

Checkpoint tốt nhất sẽ lưu tại:

- `outputs/stylecona/best.pt`
- Ảnh debug StyCona sẽ lưu tại `outputs/stylecona/stycona_debug/` (mỗi ảnh là ghép ngang: source | style_ref | augmented)

## 4) Các config chính

Trong `code/stylecona/config/default.yaml`:

- `data.root_dir`: đường dẫn dataset (mặc định `dataset`, resolve từ repo root)
- `data.layout`: `flat` hoặc `nested`
- Với `flat`: `train_split`, `val_split`, `test_split`, `target_val_split` (optional)
- Với `nested`: `source_domain`, `target_domain`, cùng các split
- `train.*`: epoch, batch size, lr, mixed precision
- `stycona.*`: bật augment và tham số decomposition
- `stycona.auxiliary_source`: `paired_styles` (file `{id}_styleK_img.png`) hoặc `batch_shuffle` (shuffle trong batch)
- `stycona.style_variant_sampling`: `random` / `first` / `cycle` — chỉ khi `paired_styles`

## 5) Smoke test

```bash
python3 train.py --config config/smoke.yaml
```

Repo tham chiếu StyCona gốc: [Senyh/StyCona](https://github.com/Senyh/StyCona)
