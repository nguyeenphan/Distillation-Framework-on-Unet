"""
StyCona-inspired augmentation
──────────────────────────────
Image-level style–content decomposition via SVD,
then style blending (σ interpolation) and optional
content mixing (u,v perturbation on top-k ranks).

Operates on numpy images (H,W,C) in [0,255] uint8.
Returns augmented image as numpy (H,W,C) float32 [0,255].
"""

import numpy as np
from typing import Optional


class StyConaAugmentor:
    """
    Given an original image x_i and a CAST auxiliary x_j:
      1. Per-channel SVD:  X_c = U_c · diag(σ_c) · V_c^T
      2. Style blend:      σ' = α · σ_i + (1−α) · σ_j
      3. Content mix:      U', V' = slerp(U_i, U_j, t)  on top-k ranks
      4. Recompose:        X'_c = U'_c · diag(σ'_c) · V'_c^T
    """

    def __init__(
        self,
        style_alpha_range: tuple[float, float] = (0.3, 0.7),
        content_mix_enabled: bool = True,
        content_t: float = 0.1,
        top_k_ranks: int = 3,
        per_channel: bool = True,
    ):
        self.style_alpha_range = style_alpha_range
        self.content_mix_enabled = content_mix_enabled
        self.content_t = content_t
        self.top_k_ranks = top_k_ranks
        self.per_channel = per_channel

    # ─── public API ──────────────────────────

    def __call__(
        self,
        image: np.ndarray,
        auxiliary: np.ndarray,
    ) -> np.ndarray:
        """
        Args:
            image:     (H, W, C) uint8 – original x_i
            auxiliary: (H, W, C) uint8 – CAST variant x_j (same content)
        Returns:
            augmented: (H, W, C) float32 clipped to [0, 255]
        """
        assert image.shape == auxiliary.shape, "image & auxiliary must have same shape"

        img_f = image.astype(np.float32)
        aux_f = auxiliary.astype(np.float32)
        alpha = np.random.uniform(*self.style_alpha_range)

        if self.per_channel:
            channels = []
            for c in range(img_f.shape[2]):
                aug_c = self._augment_channel(img_f[:, :, c], aux_f[:, :, c], alpha)
                channels.append(aug_c)
            augmented = np.stack(channels, axis=2)
        else:
            # flatten channels → single 2D matrix (H, W*C)
            H, W, C = img_f.shape
            img_2d = img_f.reshape(H, W * C)
            aux_2d = aux_f.reshape(H, W * C)
            aug_2d = self._augment_channel(img_2d, aux_2d, alpha)
            augmented = aug_2d.reshape(H, W, C)

        return np.clip(augmented, 0, 255)

    # ─── core SVD logic ─────────────────────

    def _augment_channel(
        self,
        xi: np.ndarray,
        xj: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        """
        SVD decompose → blend σ → optionally mix top-k u,v → recompose.
        xi, xj: 2D arrays (H, W) float32
        """
        # Step 1: SVD
        Ui, si, Vit = np.linalg.svd(xi, full_matrices=False)
        Uj, sj, Vjt = np.linalg.svd(xj, full_matrices=False)

        # Step 2: Style blend — interpolate singular values
        s_blended = alpha * si + (1.0 - alpha) * sj

        # Step 3: Content mix — gentle perturbation on top-k rank components
        if self.content_mix_enabled and self.top_k_ranks > 0:
            k = min(self.top_k_ranks, len(si))
            t = self.content_t
            # mix only the top-k left/right singular vectors
            U_mix = Ui.copy()
            Vt_mix = Vit.copy()
            U_mix[:, :k] = (1.0 - t) * Ui[:, :k] + t * Uj[:, :k]
            Vt_mix[:k, :] = (1.0 - t) * Vit[:k, :] + t * Vjt[:k, :]
        else:
            U_mix = Ui
            Vt_mix = Vit

        # Step 4: Recompose
        augmented = U_mix * s_blended[np.newaxis, :] @ Vt_mix
        return augmented
