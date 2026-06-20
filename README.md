# Domain-Generalizable Leaf Disease Segmentation

A PyTorch framework for pixel-level leaf disease detection that generalises to unseen visual domains (new cameras, lighting conditions, leaf varieties) by combining three complementary techniques: offline style transfer, on-the-fly style–content decomposition, and semi-supervised consistency training.

---

## Motivation

Leaf disease segmentation models trained on a single visual domain often fail when deployed in the field, where lighting, camera sensors, and plant varieties differ from training data. This framework addresses the domain gap through a three-stage pipeline that progressively increases the model's exposure to visual diversity while keeping the disease labels intact.

---

## Framework Overview

The pipeline consists of four stages executed in sequence:

**Stage A — Offline style augmentation (CAST).** A Content-Aware Style Transfer model is run once to generate a bank of re-styled copies of every training image. Each copy shares the same ground-truth segmentation mask as its source, since CAST preserves spatial structure while changing appearance. For a source image `00001`, the bank contains `00001_style0` through `00001_style8`, each paired with its own mask (`00001_style0_seg`, etc.).

**Stage B — On-the-fly style–content decomposition (StyCona-inspired).** During each training step, the current image and a randomly sampled CAST variant of the same leaf are decomposed via per-channel Singular Value Decomposition (SVD). The singular values encode *style* (colour, brightness, contrast) while the left and right singular vectors encode *content* (spatial structure). Style blending interpolates the singular values between the two images, and content mixing gently perturbs the top-k rank-one components. Recomposing the modified factors produces a novel training sample that looks different from both parents but shares their ground-truth mask.

**Stage C — Mean Teacher training.** The augmented sample is passed through two parallel augmentation paths: a strong view (heavy colour jitter, blur, random grayscale) and a weak view (minimal perturbation). The student network processes the strong view and is trained with supervised loss against the ground truth. The teacher network, an Exponential Moving Average (EMA) copy of the student, processes the weak view and produces a pseudo-label. A consistency loss forces the student's predictions on the hard view to agree with the teacher's predictions on the easy view, encouraging the model to learn domain-invariant features.

**Stage D — Loss and parameter updates.** The total loss is the sum of a supervised component (cross-entropy plus Dice loss between the student output and the ground-truth mask) and a consistency component (MSE between student and teacher soft predictions), weighted by a ramp-up schedule. The student is updated via backpropagation; the teacher is updated via EMA without gradients.

At inference time, only the student network is used — the teacher, StyCona augmentation, and CAST variants are training-time components only.

---

## Architecture

The segmentation backbone is a ResNet18-UNet: a ResNet18 encoder pretrained on ImageNet, paired with a symmetric decoder using transposed convolutions and skip connections at four resolution levels. The model outputs per-pixel class logits for background and disease.

---

## Project Structure

```
leaf-disease-dg/
├── configs/
│   └── default.yaml              # all hyperparameters in one place
│
├── data/
│   ├── dataset.py                # LeafDiseaseDataset: flat folder, auto-groups by leaf
│   └── transforms.py             # spatial augmentations (albumentations)
│
├── augmentation/
│   ├── stycona.py                # SVD decomposition → σ blend → u,v mix → recompose
│   └── view_generator.py         # strong view (student) and weak view (teacher)
│
├── models/
│   ├── resnet18_unet.py          # encoder-decoder segmentation network
│   └── ema.py                    # EMA wrapper for the teacher
│
├── losses/
│   └── losses.py                 # supervised (CE + Dice), consistency (MSE / KL)
│
├── trainers/
│   └── mean_teacher.py           # full training loop with EMA updates
│
├── utils/
│   ├── metrics.py                # IoU, Dice, precision, recall
│   └── helpers.py                # seed, checkpoint save/load, AverageMeter
│
├── scripts/
│   ├── train.py                  # entry-point: parse config → build → train
│   └── evaluate.py               # run inference on test set or unseen domains
│
└── requirements.txt
```

---

## Data Layout

Images and masks live side by side in flat folders, split into train / val / test:

```
dataset/
├── train/
│   ├── 00001_style0.png              # image
│   ├── 00001_style0_seg.png          # mask (binary: 0=bg, 255=disease)
│   ├── 00001_style1.png
│   ├── 00001_style1_seg.png
│   ├── ...
│   ├── 00001_style8.png
│   ├── 00001_style8_seg.png
│   ├── 00002_style0.png
│   ├── 00002_style0_seg.png
│   └── ...
├── val/
│   └── (same pattern)
└── test/
    └── (same pattern)
```

Naming convention:
- Image: `{content_id}_style{n}.png`
- Mask:  `{content_id}_style{n}_seg.png`

The dataset automatically groups images by `content_id` (e.g. `00001`), so StyCona can sample a random auxiliary from the same leaf but a different style. All styles of the same leaf share the same ground-truth disease regions.

---

## Installation

```bash
git clone <repo-url> && cd leaf-disease-dg
pip install -r requirements.txt
```

Key dependencies: PyTorch >= 2.0, torchvision, albumentations, OpenCV, PyYAML.

---

## Usage

### Train

```bash
python scripts/train.py --config configs/default.yaml
```

The best checkpoint (by validation Dice) is saved to `checkpoints/best.pth`.

### Evaluate

```bash
# Using test set from config
python scripts/evaluate.py \
    --config     configs/default.yaml \
    --checkpoint checkpoints/best.pth

# Or a custom unseen-domain folder
python scripts/evaluate.py \
    --config     configs/default.yaml \
    --checkpoint checkpoints/best.pth \
    --test_dir   ./unseen_domain
```

Reports IoU, Dice, precision, and recall on the test set.

---

## Key Hyperparameters

All hyperparameters are centralised in `configs/default.yaml`:

| Parameter | Location | Default | Notes |
|-----------|----------|---------|-------|
| `stycona.style_alpha_range` | StyCona | [0.3, 0.7] | How far style blending deviates from the original |
| `stycona.content_mix.enabled` | StyCona | true | Set false to ablate content mixing (style-only) |
| `stycona.content_mix.t` | StyCona | 0.1 | Content blend strength — keep small to preserve labels |
| `stycona.content_mix.top_k_ranks` | StyCona | 3 | Only mix the top-k singular components |
| `mean_teacher.ema_decay` | Teacher | 0.999 | Higher = more stable teacher, slower adaptation |
| `mean_teacher.consistency_rampup_epochs` | Teacher | 30 | Linear ramp-up from 0 → max consistency weight |
| `mean_teacher.consistency_weight_max` | Teacher | 1.0 | Final consistency loss weight |
| `views.strong.color_jitter` | Views | 0.4 | Colour augmentation strength for student |
| `loss.consistency.type` | Loss | "mse" | "mse" or "kl" divergence |

---

## Design Decisions

**CAST auxiliary selection.** The CAST auxiliary for StyCona is sampled from the same content ID (same leaf, different style), not from a different leaf. This ensures the ground-truth mask remains valid after SVD decomposition and recomposition, which is critical when disease spots are small.

**StyCona at image-level.** SVD is applied directly to image pixels (per channel) rather than to intermediate feature maps. This avoids the numerical instability of differentiable SVD in the backward pass and eliminates the computational overhead of running SVD inside the training graph.

**Content mixing as the differentiator.** Since CAST already performs style augmentation, the primary value-add of StyCona in this pipeline is content mixing (perturbing the spatial singular vectors). This can be ablated by setting `stycona.content_mix.enabled: false`.

**Mean Teacher over knowledge distillation.** The teacher is an EMA copy of the student, not a separately trained model. The consistency loss directly encourages domain-invariant representations by forcing agreement between differently-augmented views of the same image.

---

## Ablation Experiments

| Experiment | Configuration change |
|------------|---------------------|
| Baseline (no StyCona) | `stycona.enabled: false` |
| Style-only (no content mix) | `stycona.content_mix.enabled: false` |
| No consistency loss | `mean_teacher.consistency_weight_max: 0` |
| Different consistency loss | `loss.consistency.type: "kl"` |

---

## Training Pipeline (per step)

```
┌─────────────────────────────────────────────────────────────┐
│  Sample batch from train/:                                  │
│    image    = 00001_style3.png                               │
│    mask     = 00001_style3_seg.png                           │
│    auxiliary = 00001_style7.png  (random, same leaf)         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  StyCona: SVD(image) & SVD(auxiliary)                       │
│    → blend σ' = α·σᵢ + (1−α)·σⱼ                           │
│    → mix top-k u,v:  U' = (1−t)·Uᵢ + t·Uⱼ                │
│    → recompose x' = U'·diag(σ')·V'ᵀ                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
              ┌────────┴────────┐
              ▼                 ▼
    ┌──────────────┐   ┌──────────────┐
    │  Strong view │   │  Weak view   │
    │  (jitter,    │   │  (near-      │
    │   blur,      │   │   original)  │
    │   grayscale) │   │              │
    └──────┬───────┘   └──────┬───────┘
           │                  │
           ▼                  ▼
    ┌──────────────┐   ┌──────────────┐
    │   Student    │   │   Teacher    │
    │   fθ        │   │   fξ (EMA)  │
    └──────┬───────┘   └──────┬───────┘
           │                  │
           ▼                  ▼
    ┌──────────────────────────────────┐
    │  L = L_sup(pred, mask)           │
    │    + λ(t) · L_cons(pred, t_pred)│
    └──────────────────────────────────┘
           │
    ┌──────┴───────┐
    │  Backprop θ  │  →  EMA update ξ
    └──────────────┘
```

---

## Evaluation Metrics

- **IoU (Intersection over Union):** area of overlap divided by area of union between prediction and ground truth.
- **Dice coefficient (F1):** harmonic mean of precision and recall.
- **Precision:** fraction of predicted disease pixels that are truly diseased.
- **Recall:** fraction of truly diseased pixels that were correctly detected.

---

## References

- **CAST:** Y. Zhang et al., "Domain Enhanced Arbitrary Image Style Transfer via Contrastive Learning," SIGGRAPH 2022.
- **StyCona:** S. Li et al., "StyCona: Style-based Data Augmentation via SVD Style–Content Decomposition for Domain Generalisation."
- **Mean Teacher:** A. Tarvainen and H. Valpola, "Mean teachers are better role models," NeurIPS 2017.

---

## License

This project is for research purposes.
