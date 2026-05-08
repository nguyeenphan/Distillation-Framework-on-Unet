import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import dice_score


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_dice = 0.0
    steps = 0

    for images, masks in dataloader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        total_dice += dice_score(logits, masks)
        steps += 1

    mean_dice = total_dice / max(steps, 1)
    return {"dice": mean_dice}
