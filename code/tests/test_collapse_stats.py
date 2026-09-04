"""Self-check: the two collapse detectors must fire on the two collapse modes."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import math
import torch
from utils.metrics import collapse_stats


def test_collapse_stats():
    B, H, W = 2, 8, 8
    LN2 = math.log(2)

    # Healthy: confident, mixed prediction.
    logits = torch.full((B, 2, H, W), -5.0)
    logits[:, 1, :, :4] = 5.0          # left half = foreground
    logits[:, 0, :, 4:] = 5.0
    ent, fg = collapse_stats(logits)
    assert ent < 0.05 * LN2, ent       # near-zero entropy
    assert abs(fg - 0.5) < 1e-6, fg

    # Collapse (a): flat-uncertain — entropy saturates at ln(C).
    ent, fg = collapse_stats(torch.zeros(B, 2, H, W))
    assert abs(ent - LN2) < 1e-5, ent

    # Collapse (b): all-background, confident — LOW entropy, fg -> 0.
    # This is the mode entropy alone would miss.
    logits = torch.full((B, 2, H, W), -5.0)
    logits[:, 0] = 5.0
    ent, fg = collapse_stats(logits)
    assert ent < 0.05 * LN2, ent
    assert fg == 0.0, fg


if __name__ == "__main__":
    test_collapse_stats()
    print("collapse_stats OK")
