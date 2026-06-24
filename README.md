# Domain-Generalizable Leaf Disease Segmentation

A PyTorch framework for pixel-level leaf disease detection that generalises to unseen visual domains (new cameras, lighting conditions, leaf varieties) by combining three complementary techniques: offline style transfer (CAST), on-the-fly style–content decomposition (StyCona), and semi-supervised consistency training (Mean Teacher).

---

## Motivation

Leaf disease segmentation models trained on a single visual domain often fail when deployed in the field, where lighting, camera sensors, and plant varieties differ from training data. This framework addresses the domain gap through a three-stage pipeline that progressively increases the model's exposure to visual diversity while keeping the disease labels intact.

---

## Framework Overview

**Stage A — Offline style augmentation (CAST).** A Content-Aware Style Transfer model is run once to generate a bank of re-styled copies of every training image. Each copy shares the same ground-truth segmentation mask as its source, since CAST preserves spatial structure while changing appearance. For a source image `00001`, the bank contains `00001_style0` through `00001_style8`, each paired with its own mask.

**Stage B — On-the-fly style–content decomposition (StyCona).** During each training step, the current image and a randomly sampled CAST variant of the same leaf are decomposed via per-channel SVD. The singular values encode *style* (colour, brightness, contrast) while the singular vectors encode *content* (spatial structure). Style blending interpolates the singular values; content mixing gently perturbs the top-k rank-one components. Recomposing the modified factors produces a novel sample that looks different from both parents but shares their ground-truth mask.

**Stage C — Mean Teacher training.** The StyCona-augmented sample is passed through two augmentation paths: a strong view (heavy colour jitter, blur, random grayscale) fed to the Student, and a weak view (minimal perturbation) fed to the Teacher. The Teacher is an EMA copy of the Student and produces a pseudo-label. A consistency loss forces the Student to agree with the Teacher across different views, encouraging domain-invariant features.

**Stage D — Loss and parameter updates.** Total loss = supervised (Focal + Dice + L1) + λ(t) × consistency (MSE or KL), where λ ramps up linearly over the first N epochs. The Student is updated via backpropagation; the Teacher via EMA without gradients. Only the Student is used at inference time.

---

## Architecture

**Backbone:** ResNet18-UNet — ResNet18 encoder pretrained on ImageNet, symmetric decoder with transposed convolutions and skip connections at four resolution levels.

**Head:** ASPP (Atrous Spatial Pyramid Pooling) with dilation rates [6, 12, 18] for multi-scale context, followed by two residual refinement blocks and a 1×1 classifier. Outputs per-pixel class logits for background and disease.

**Total parameters:** ~14.6M

---

## Project Structure

```
Distillation-Framework-on-Unet/
├── dataset/
│   ├── train/
│   ├── val/
│   └── test/
│
└── code/
    ├── configs/
    │   └── default.yaml              # all hyperparameters in one place
    │
    ├── data/
    │   ├── dataset.py                # LeafDiseaseDataset: flat folder, groups by leaf
    │   └── transforms.py             # spatial augmentations (albumentations)
    │
    ├── augmentation/
    │   ├── stycona.py                # SVD decompose → σ blend → U,V mix → recompose
    │   └── view_generator.py         # strong view (student) and weak view (teacher)
    │
    ├── models/
    │   ├── resnet18_unet.py          # ResNet18 encoder + UNet decoder + ASPP head
    │   └── ema.py                    # EMA wrapper for the teacher network
    │
    ├── losses/
    │   └── losses.py                 # Focal, Dice, L1 (supervised) + MSE/KL (consistency)
    │
    ├── trainers/
    │   └── mean_teacher.py           # full training loop with per-epoch CSV + TensorBoard log
    │
    ├── utils/
    │   ├── metrics.py                # IoU, Dice, precision, recall
    │   └── helpers.py                # seed, checkpoint save/load, AverageMeter
    │
    ├── scripts/
    │   ├── train.py                  # entry-point: parse config → build → train
    │   ├── evaluate.py               # inference on test set with optional visual output
    │   ├── visualize_stycona.py      # quick 4-panel StyCona preview
    │   └── stycona_cases.py          # full 6-row case breakdown of StyCona pipeline
    │
    ├── checkpoints/
    │   └── best.pth                  # best student checkpoint (by val Dice)
    │
    └── logs/
        ├── train_log.csv             # epoch, train_loss, val_dice, val_iou
        └── events.out.tfevents.*     # TensorBoard log
```

---

## Data Layout

```
dataset/
├── train/
│   ├── 00001_img.png              # base image
│   ├── 00001_seg.png              # mask (0 = background, 1 = disease)
│   ├── 00001_style0_img.png       # CAST style variant 0
│   ├── 00001_style0_seg.png       # same mask as base
│   ├── 00001_style1_img.png
│   ├── 00001_style1_seg.png
│   └── ...
├── val/
└── test/
```

Naming convention:
- Base image:  `{content_id}_img.png`
- Base mask:   `{content_id}_seg.png`
- Style image: `{content_id}_style{n}_img.png`
- Style mask:  `{content_id}_style{n}_seg.png`

The dataset auto-groups images by `content_id` so StyCona can sample a random CAST auxiliary from the same leaf. With `base_image_only: false`, all style variants become supervised training samples with their own masks.

---

## Installation

```bash
# Requires Python 3.10+ and a native arm64 environment on Apple Silicon
git clone <repo-url>
cd Distillation-Framework-on-Unet/code

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Key dependencies: `torch>=2.0`, `torchvision`, `albumentations>=2.0`, `opencv-python`, `PyYAML`, `tqdm`, `tensorboard`.

---

## Usage

### Train

```bash
cd code
.venv/bin/python3 scripts/train.py --config configs/default.yaml
```

**Outputs:**
- `checkpoints/best.pth` — best student model by validation Dice
- `logs/train_log.csv` — per-epoch `train_loss`, `val_dice`, `val_iou`
- `logs/events.out.tfevents.*` — TensorBoard log

Monitor training with TensorBoard:
```bash
.venv/bin/tensorboard --logdir logs/
```

### Evaluate

```bash
# All images (base + style variants)
.venv/bin/python3 scripts/evaluate.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/best.pth \
    --vis_dir ./visualizations

# Base images only
.venv/bin/python3 scripts/evaluate.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/best.pth \
    --base_only
```

The `--vis_dir` flag saves a side-by-side PNG per image: **Original | Ground Truth | Prediction**, with disease regions highlighted in red and per-image Dice score in the title.

### Visualize StyCona pipeline

```bash
# Quick 4-panel preview (original | auxiliary | StyCona | mask)
.venv/bin/python3 scripts/visualize_stycona.py --config configs/default.yaml --n 6

# Full 6-row case breakdown
.venv/bin/python3 scripts/stycona_cases.py --config configs/default.yaml --n 4 --out viz/cases
```

The full breakdown shows:
| Row | Content |
|-----|---------|
| 1 | Original · GT mask overlay · Auxiliary (CAST) |
| 2 | Style blend only at α = 0.30 / 0.50 / 0.70 |
| 3 | Full StyCona (style + content mix) at α = 0.30 / 0.50 / 0.70 |
| 4 | StyCona output · Strong view (Student) · Weak view (Teacher) |
| 5 | All CAST style variants for this leaf |
| 6 | StyCona xi + each style variant (α = 0.50) |

---

## Key Hyperparameters

All hyperparameters are in `configs/default.yaml`:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `data.base_image_only` | `false` | `true` = supervised on base only (472 samples); `false` = all styles (4248 samples) |
| `stycona.style_alpha_range` | [0.3, 0.7] | Range of σ interpolation between xi and xj |
| `stycona.content_mix.enabled` | `true` | Set `false` to ablate content mixing (style-only) |
| `stycona.content_mix.t` | 0.1 | Content blend strength — keep small to preserve labels |
| `stycona.content_mix.top_k_ranks` | 3 | Number of top singular components to mix |
| `mean_teacher.ema_decay` | 0.999 | Higher = more stable teacher, slower adaptation |
| `mean_teacher.consistency_rampup_epochs` | 30 | Linear ramp-up from 0 to max consistency weight |
| `mean_teacher.consistency_weight_max` | 1.0 | Final consistency loss weight |
| `loss.supervised.focal_gamma` | 2.0 | Focal loss focusing parameter (0 = standard CE) |
| `loss.consistency.type` | `"mse"` | `"mse"` or `"kl"` divergence |
| `training.epochs` | 150 | |
| `training.lr` | 5e-4 | |
| `training.warmup_epochs` | 10 | Linear LR warmup before cosine schedule |

---

## Training Results (Run 1 — base_image_only: true, 100 epochs)

| Epoch | Val Dice | Val IoU |
|-------|----------|---------|
| 5 | 0.575 | 0.404 |
| 35 | 0.609 | 0.438 |
| 55 | 0.686 | 0.522 |
| 70 | 0.703 | 0.541 |
| **95** | **0.711** | **0.551** |
| 100 | 0.708 | 0.548 |

Best checkpoint at epoch 95: **Dice = 0.711, IoU = 0.551**

---

## Ablation Experiments

| Experiment | Config change |
|------------|---------------|
| Baseline (no StyCona) | `stycona.enabled: false` |
| Style-only (no content mix) | `stycona.content_mix.enabled: false` |
| No consistency loss | `mean_teacher.consistency_weight_max: 0` |
| KL consistency | `loss.consistency.type: "kl"` |
| Base images only | `data.base_image_only: true` |

---

## Training Pipeline (per step)

```
Sample batch:
  image     = 00001_img.png          (supervised sample)
  mask      = 00001_seg.png
  auxiliary = 00001_style3_img.png   (CAST donor, same leaf)
        |
        v
  StyCona augmentation:
    SVD(image) & SVD(auxiliary)
    -> blend  sigma' = alpha*si + (1-alpha)*sj
    -> mix    U'[:, :k] = (1-t)*Ui + t*Uj
    -> recompose  x' = U' * diag(sigma') * Vt'
        |
      split
       / \
      /   \
Strong     Weak
(jitter,   (near-
 blur,      original)
 gray)
  |              |
Student        Teacher
  |              | (EMA, no grad)
  +-----> L_sup(pred, mask)
  +-----> L_cons(student_pred, teacher_pred) * lambda(epoch)
        |
    backprop -> update Student theta
    EMA      -> update Teacher xi
```

---

## Evaluation Metrics

- **Dice (F1):** harmonic mean of precision and recall — primary metric
- **IoU:** intersection over union between prediction and ground truth
- **Precision:** fraction of predicted disease pixels that are truly diseased
- **Recall:** fraction of truly diseased pixels correctly detected

---

## References

- **CAST:** Y. Zhang et al., "Domain Enhanced Arbitrary Image Style Transfer via Contrastive Learning," SIGGRAPH 2022.
- **StyCona:** S. Li et al., "StyCona: Style-based Data Augmentation via SVD Style–Content Decomposition for Domain Generalisation."
- **Mean Teacher:** A. Tarvainen and H. Valpola, "Mean teachers are better role models," NeurIPS 2017.
- **DeSTSeg:** Z. Zhang et al., "DeSTSeg: Segmentation Guided Denoising Student-Teacher for Anomaly Detection," CVPR 2023.

---

## License

This project is for research purposes.
