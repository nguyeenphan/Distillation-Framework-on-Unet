"""
Exponential Moving Average (EMA) Model
───────────────────────────────────────
Wraps the Teacher network: ξ ← m·ξ + (1−m)·θ

Usage:
    teacher = EMAModel(student, decay=0.999)
    # after each student update:
    teacher.update(student)
    # inference:
    pred = teacher(x)
"""

import copy
import torch
import torch.nn as nn


class EMAModel(nn.Module):
    """Teacher = exponential moving average of the Student."""

    def __init__(self, student: nn.Module, decay: float = 0.999):
        super().__init__()
        self.model = copy.deepcopy(student)
        self.decay = decay

        # Teacher does not require gradients
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, student: nn.Module) -> None:
        """Update teacher params: ξ ← m·ξ + (1−m)·θ"""
        m = self.decay
        for p_teacher, p_student in zip(
            self.model.parameters(), student.parameters()
        ):
            p_teacher.data.mul_(m).add_(p_student.data, alpha=1.0 - m)

        # Also update buffers (batch-norm running stats)
        for b_teacher, b_student in zip(
            self.model.buffers(), student.buffers()
        ):
            b_teacher.data.copy_(b_student.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
