"""
Losses
──────
Supervised:   L_sup  = Focal + Dice + L1  (student output vs GT mask)
Consistency:  L_cons = MSE or KL          (student output vs teacher pseudo-label)
Total:        L = L_sup + λ(t) · L_cons

Focal loss replaces standard CE to handle the severe class imbalance
in leaf disease segmentation (small disease spots, large background).
Inspired by DeSTSeg (Zhang et al., 2023).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Focal Loss ────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal loss for dense classification (Lin et al., 2017).

    Downweights easy-to-classify pixels (mostly background) so the model
    focuses on hard pixels near disease boundaries.

    L_focal = -α_t · (1 - p_t)^γ · log(p_t)

    Args:
        gamma:  focusing parameter. γ=0 → standard CE. γ=2 is typical.
        alpha:  class weight for the positive (disease) class. None = no weighting.
    """

    def __init__(self, gamma: float = 2.0, alpha: float | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  (B, C, H, W) raw model output
        targets: (B, H, W) long class indices
        """
        num_classes = logits.shape[1]

        # Softmax probabilities
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)

        # One-hot targets → (B, C, H, W)
        targets_oh = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        # p_t = probability of the true class
        p_t = (probs * targets_oh).sum(dim=1)  # (B, H, W)
        p_t = p_t.clamp(min=1e-7)  # numerical stability

        # Focal weight
        focal_weight = (1.0 - p_t) ** self.gamma

        # CE component
        ce = -torch.log(p_t)

        # Alpha weighting (optional)
        if self.alpha is not None:
            # α for disease class (1), (1-α) for background (0)
            alpha_t = targets.float() * self.alpha + (1.0 - targets.float()) * (1.0 - self.alpha)
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce
        return loss.mean()


# ── Dice Loss ─────────────────────────────────

class DiceLoss(nn.Module):
    """Soft Dice loss for binary / multi-class segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  (B, C, H, W) raw model output
        targets: (B, H, W) long class indices
        """
        probs = F.softmax(logits, dim=1)
        num_classes = logits.shape[1]

        targets_oh = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = (probs * targets_oh).sum(dims)
        cardinality = probs.sum(dims) + targets_oh.sum(dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


# ── L1 Segmentation Loss ─────────────────────

class SegL1Loss(nn.Module):
    """
    L1 loss between predicted probabilities and binary mask.
    Encourages sharper segmentation boundaries (DeSTSeg Eq. 6).
    """

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  (B, C, H, W) raw model output
        targets: (B, H, W) long class indices
        """
        # Use disease-class probability
        probs = F.softmax(logits, dim=1)[:, 1, :, :]  # (B, H, W)
        targets_f = targets.float()
        return F.l1_loss(probs, targets_f)


# ── Supervised Loss ───────────────────────────

class SupervisedLoss(nn.Module):
    """
    Focal + Dice + L1.

    Focal replaces CE for better handling of class imbalance.
    Dice ensures overlap-based optimisation.
    L1 sharpens segmentation boundaries.
    """

    def __init__(
        self,
        focal_weight: float = 1.0,
        dice_weight: float = 1.0,
        l1_weight: float = 0.5,
        focal_gamma: float = 2.0,
        focal_alpha: float | None = None,
    ):
        super().__init__()
        self.focal = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        self.dice = DiceLoss()
        self.l1 = SegL1Loss()
        self.focal_w = focal_weight
        self.dice_w = dice_weight
        self.l1_w = l1_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        l_focal = self.focal(logits, targets)
        l_dice = self.dice(logits, targets)
        l_l1 = self.l1(logits, targets)
        return self.focal_w * l_focal + self.dice_w * l_dice + self.l1_w * l_l1


# ── Consistency Loss ──────────────────────────

class ConsistencyLoss(nn.Module):
    """MSE or KL between student and teacher soft predictions."""

    def __init__(self, loss_type: str = "mse"):
        super().__init__()
        assert loss_type in ("mse", "kl"), f"Unknown consistency type: {loss_type}"
        self.loss_type = loss_type

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        s_probs = F.softmax(student_logits, dim=1)
        t_probs = F.softmax(teacher_logits.detach(), dim=1)

        if self.loss_type == "mse":
            return F.mse_loss(s_probs, t_probs)
        else:
            s_log = F.log_softmax(student_logits, dim=1)
            return F.kl_div(s_log, t_probs, reduction="batchmean")


# ── Consistency weight ramp-up ────────────────

def consistency_weight(epoch: int, max_weight: float, rampup_epochs: int) -> float:
    """Linear ramp-up: 0 → max_weight over rampup_epochs."""
    if rampup_epochs == 0:
        return max_weight
    return max_weight * min(1.0, epoch / rampup_epochs)
