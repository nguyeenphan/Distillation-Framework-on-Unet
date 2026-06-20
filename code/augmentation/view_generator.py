"""
View Generator
───────────────
Takes a StyCona-augmented image and produces:
  - strong_view : heavy colour/blur augmentations → fed to Student
  - weak_view   : minimal augmentations           → fed to Teacher
"""

import numpy as np
import cv2
import torch


class ViewGenerator:
    """Create strong and weak views from an augmented image tensor."""

    def __init__(self, strong_cfg: dict, weak_cfg: dict):
        self.strong_cfg = strong_cfg
        self.weak_cfg = weak_cfg

    def __call__(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image: (C, H, W) float32 tensor in [0, 1]
        Returns:
            strong_view, weak_view: same shape tensors
        """
        strong_view = self._apply_augmentation(image, self.strong_cfg)
        weak_view = self._apply_augmentation(image, self.weak_cfg)
        return strong_view, weak_view

    def _apply_augmentation(
        self, image: torch.Tensor, cfg: dict
    ) -> torch.Tensor:
        """Apply colour-level augmentations (no spatial — mask must stay aligned)."""
        x = image.clone()

        # Colour jitter
        jitter = cfg.get("color_jitter", 0.0)
        if jitter > 0:
            x = self._color_jitter(x, strength=jitter)

        # Random grayscale
        if np.random.random() < cfg.get("random_gray_prob", 0.0):
            x = self._to_grayscale(x)

        # Gaussian blur
        if np.random.random() < cfg.get("gaussian_blur_prob", 0.0):
            x = self._gaussian_blur(x)

        return x

    # ── simple augmentation ops ──────────────

    @staticmethod
    def _color_jitter(x: torch.Tensor, strength: float) -> torch.Tensor:
        """Random brightness, contrast, saturation shift."""
        # Brightness
        x = x + (torch.rand(1).item() - 0.5) * 2 * strength * 0.3
        # Contrast
        mean = x.mean()
        x = (x - mean) * (1.0 + (torch.rand(1).item() - 0.5) * 2 * strength * 0.3) + mean
        return x.clamp(0, 1)

    @staticmethod
    def _to_grayscale(x: torch.Tensor) -> torch.Tensor:
        """Convert to grayscale but keep 3 channels."""
        gray = 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]
        return gray.unsqueeze(0).expand(3, -1, -1)

    @staticmethod
    def _gaussian_blur(x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        """Simple Gaussian blur via OpenCV."""
        np_img = (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(np_img, (kernel_size, kernel_size), 0)
        return torch.from_numpy(blurred.transpose(2, 0, 1)).float().div(255.0)
