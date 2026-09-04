"""
Metrics
───────
IoU (Jaccard), Dice (F1), Precision, Recall for segmentation evaluation.
"""

import torch
import numpy as np


def compute_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 2,
) -> dict:
    """
    Args:
        preds:   (N, H, W) predicted class indices
        targets: (N, H, W) ground-truth class indices
    Returns:
        dict with iou, dice, precision, recall (averaged over foreground classes)
    """
    eps = 1e-7
    ious, dices, precisions, recalls = [], [], [], []

    for cls in range(1, num_classes):  # skip background (cls=0)
        pred_mask = (preds == cls)
        true_mask = (targets == cls)

        tp = (pred_mask & true_mask).sum().float()
        fp = (pred_mask & ~true_mask).sum().float()
        fn = (~pred_mask & true_mask).sum().float()

        iou = tp / (tp + fp + fn + eps)
        dice = 2 * tp / (2 * tp + fp + fn + eps)
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)

        ious.append(iou.item())
        dices.append(dice.item())
        precisions.append(prec.item())
        recalls.append(rec.item())

    return {
        "iou": np.mean(ious),
        "dice": np.mean(dices),
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
    }


def collapse_stats(logits: torch.Tensor) -> tuple[float, float]:
    """Two collapse detectors for a consistency-regularised segmenter.

    Returns (mean per-pixel entropy, predicted foreground rate).
    Entropy rising toward ln(C) => flat-uncertain collapse.
    Foreground rate falling toward 0 => all-background collapse.
    """
    probs = torch.softmax(logits, dim=1)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(1).mean()
    fg_rate = (logits.argmax(1) == 1).float().mean()
    return entropy.item(), fg_rate.item()
