import random

import numpy as np
import torch
from torch import Tensor


class StyConTransform(torch.nn.Module):
    """
    Faithful implementation based on official StyCona repository logic.
    """

    def __init__(self, p: float = 1.0, min_step: int = 1, min_start: int = 1, k_vectors: int = 16) -> None:
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in range [0, 1]")
        if min_step < 1:
            raise ValueError("min_step must be >= 1")
        if min_start < 0:
            raise ValueError("min_start must be >= 0")
        if k_vectors < 1:
            raise ValueError("k_vectors must be >= 1")
        self.p = p
        self.min_step = min_step
        self.min_start = min_start
        self.k_vectors = k_vectors

    def _random_indices(self, width: int) -> tuple[list[int], list[int]]:
        src_idx = random.choices(range(width), k=self.k_vectors)
        ref_idx = random.choices(range(width), k=self.k_vectors)
        return src_idx, ref_idx

    def forward(self, tensor: Tensor, tensora: Tensor) -> Tensor:
        if torch.rand(1, device=tensor.device) >= self.p:
            return tensor

        if tensor.ndim != 3 or tensora.ndim != 3:
            raise ValueError("StyConTransform expects tensors with shape [C, H, W]")
        if tensor.shape != tensora.shape:
            raise ValueError("source tensor and style tensor must have the same shape")

        # tensor/tensora: [C, H, W]
        u, s, vh = torch.linalg.svd(tensor, full_matrices=False)
        ua, sa, vha = torch.linalg.svd(tensora, full_matrices=False)

        width = s.shape[-1]
        half_width = max(width // 2, 2)

        for channel_idx in range(s.shape[0]):
            # Style mix on singular values (element-wise random blend)
            c = torch.rand(width, device=s.device)
            s[channel_idx] = c * s[channel_idx] + (1.0 - c) * sa[channel_idx]

            # Randomly drop a stride pattern in S to perturb style spectrum
            start = self.min_start + int(torch.randint(low=0, high=half_width, size=(1,), device=s.device).item())
            if torch.rand(1, device=s.device) < 0.5:
                step = int(torch.randint(low=self.min_step, high=half_width, size=(1,), device=s.device).item())
                step = max(step, 1)
                s[channel_idx, start::step] = 0.0

            # Content mix on U columns
            a = float(np.random.beta(1, 1))
            src_idx, ref_idx = self._random_indices(width)
            u[channel_idx, :, src_idx] = a * u[channel_idx, :, src_idx] + (1.0 - a) * ua[channel_idx, :, ref_idx]

            # Content mix on Vh rows
            b = float(np.random.beta(1, 1))
            src_idx, ref_idx = self._random_indices(width)
            vh[channel_idx, src_idx, :] = b * vh[channel_idx, src_idx, :] + (1.0 - b) * vha[channel_idx, ref_idx, :]

        mixed = u @ torch.diag_embed(s) @ vh
        return mixed.clamp(0.0, 1.0)
