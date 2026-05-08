import argparse
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image

from config.augment import StyConTransform
from config.decompose import ExperimentConfig
from config.pipeline import load_experiment_config
from data.loader import build_dataloader
from evaluation.evaluator import evaluate
from losses.losses import segmentation_loss
from models.resnet18_unet import ResNetUNet


@dataclass(frozen=True)
class StyConaRuntime:
    prob: float
    transform: StyConTransform
    auxiliary_source: str


@dataclass(frozen=True)
class DataLoaders:
    train: DataLoader
    val_source: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    val_target: DataLoader[tuple[torch.Tensor, torch.Tensor]]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def resolve_path(path_value: str) -> Path:
    candidate = Path(path_value)
    if candidate.exists():
        return candidate.resolve()

    project_root = Path(__file__).resolve().parents[2]
    fallback = project_root / path_value
    if fallback.exists():
        return fallback.resolve()
    return fallback


def maybe_apply_stycona_paired(
    images: torch.Tensor,
    style_images: torch.Tensor,
    transform: StyConTransform,
    prob: float,
) -> torch.Tensor:
    if random.random() > prob:
        return images
    mixed = [transform(images[idx], style_images[idx]) for idx in range(images.size(0))]
    return torch.stack(mixed, dim=0)


def maybe_apply_stycona_batch_shuffle(images: torch.Tensor, transform: StyConTransform, prob: float) -> torch.Tensor:
    if random.random() > prob or images.size(0) < 2:
        return images

    shuffled = images[torch.randperm(images.size(0), device=images.device)]
    mixed = [transform(source, style) for source, style in zip(images, shuffled)]
    return torch.stack(mixed, dim=0)


def build_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train StyCona segmentation baseline")
    parser.add_argument("--config", type=str, default=None, help="Path to yaml config")
    parser.add_argument(
        "--stycona-debug-save",
        action="store_true",
        help="Save before/style/after StyCona visualizations each epoch",
    )
    parser.add_argument(
        "--stycona-debug-max-samples",
        type=int,
        default=4,
        help="Maximum debug samples saved per epoch",
    )
    return parser.parse_args()


def build_model_and_optimizer(
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[nn.Module, torch.optim.Optimizer, torch.amp.GradScaler]:
    model = ResNetUNet(n_class=1).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.train.mixed_precision and device.type == "cuda")
    return model, optimizer, scaler


def resolve_split_paths(config: ExperimentConfig, root_dir: Path) -> tuple[Path, Path, Path]:
    data = config.data
    if data.layout == "flat":
        train_dir = root_dir / data.train_split
        val_dir = root_dir / data.val_split
        target_val_name = data.target_val_split or data.val_split
        target_val_dir = root_dir / target_val_name
        return train_dir, val_dir, target_val_dir

    source_root = root_dir / data.source_domain
    target_root = root_dir / data.target_domain
    train_dir = source_root / data.train_split
    val_source_dir = source_root / data.val_split
    val_target_dir = target_root / data.val_split
    return train_dir, val_source_dir, val_target_dir


def build_all_dataloaders(config: ExperimentConfig) -> DataLoaders:
    root_dir = resolve_path(config.data.root_dir)
    train_dir, val_source_dir, val_target_dir = resolve_split_paths(config, root_dir)

    common_kwargs = {
        "image_size": config.data.image_size,
        "batch_size": config.train.batch_size,
        "num_workers": config.data.num_workers,
    }

    use_paired_train = config.stycona.enable and config.stycona.auxiliary_source == "paired_styles"
    train_loader = build_dataloader(
        split_dir=str(train_dir),
        shuffle=True,
        paired_style=use_paired_train,
        style_variant_sampling=config.stycona.style_variant_sampling,
        **common_kwargs,
    )
    val_source_loader = build_dataloader(
        split_dir=str(val_source_dir),
        shuffle=False,
        **common_kwargs,
    )
    if val_source_dir.resolve() == val_target_dir.resolve():
        val_target_loader = val_source_loader
    else:
        val_target_loader = build_dataloader(
            split_dir=str(val_target_dir),
            shuffle=False,
            **common_kwargs,
        )
    return DataLoaders(train=train_loader, val_source=val_source_loader, val_target=val_target_loader)


def build_stycona_runtime(config: ExperimentConfig, device: torch.device) -> Optional[StyConaRuntime]:
    if not config.stycona.enable:
        return None
    transform = StyConTransform(
        p=1.0,
        min_step=config.stycona.min_step,
        min_start=config.stycona.min_start,
        k_vectors=config.stycona.k_vectors,
    ).to(device)
    return StyConaRuntime(
        prob=config.stycona.prob,
        transform=transform,
        auxiliary_source=config.stycona.auxiliary_source,
    )


@torch.no_grad()
def save_stycona_debug_samples(
    images: torch.Tensor,
    transform: StyConTransform,
    epoch: int,
    save_dir: Path,
    max_samples: int,
    style_images: Optional[torch.Tensor] = None,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    sample_count = min(max_samples, images.size(0))

    if style_images is not None:
        for sample_idx in range(sample_count):
            source = images[sample_idx]
            style_ref = style_images[sample_idx]
            augmented = transform(source, style_ref)
            comparison = torch.cat([source, style_ref, augmented], dim=2).detach().cpu()
            save_image(comparison, save_dir / f"epoch_{epoch:03d}_sample_{sample_idx:02d}.png")
        return

    if images.size(0) < 2:
        return

    shuffled = images[torch.randperm(images.size(0), device=images.device)]
    for sample_idx in range(sample_count):
        source = images[sample_idx]
        style_ref = shuffled[sample_idx]
        augmented = transform(source, style_ref)
        comparison = torch.cat([source, style_ref, augmented], dim=2).detach().cpu()
        save_image(comparison, save_dir / f"epoch_{epoch:03d}_sample_{sample_idx:02d}.png")


def run_train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    stycona_runtime: Optional[StyConaRuntime],
    log_interval: int,
) -> float:
    model.train()
    running_loss = 0.0

    iterator = tqdm(enumerate(dataloader), total=len(dataloader), desc="Train", leave=False)
    for step, batch in iterator:
        if len(batch) == 3:
            images, masks, style_images = batch
            style_images = style_images.to(device, non_blocking=True)
        else:
            images, masks = batch
            style_images = None

        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if stycona_runtime is not None:
            if stycona_runtime.auxiliary_source == "paired_styles" and style_images is not None:
                images = maybe_apply_stycona_paired(
                    images,
                    style_images,
                    transform=stycona_runtime.transform,
                    prob=stycona_runtime.prob,
                )
            elif stycona_runtime.auxiliary_source == "batch_shuffle":
                images = maybe_apply_stycona_batch_shuffle(
                    images,
                    transform=stycona_runtime.transform,
                    prob=stycona_runtime.prob,
                )

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = segmentation_loss(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.item())
        if step % max(log_interval, 1) == 0:
            iterator.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(len(dataloader), 1)


def main() -> None:
    args = build_cli_args()

    config = load_experiment_config(args.config)
    set_seed(config.train.seed)
    device = resolve_device(config.runtime.device)

    model, optimizer, scaler = build_model_and_optimizer(config=config, device=device)
    loaders = build_all_dataloaders(config=config)
    stycona_runtime = build_stycona_runtime(config=config, device=device)

    output_dir = resolve_path(config.runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / config.runtime.checkpoint_name
    debug_dir = output_dir / "stycona_debug"

    best_target_dice = -1.0
    for epoch in range(1, config.train.epochs + 1):
        if args.stycona_debug_save and stycona_runtime is not None:
            debug_batch = next(iter(loaders.train))
            if len(debug_batch) == 3:
                debug_images, _, debug_styles = debug_batch
                debug_images = debug_images.to(device, non_blocking=True)
                debug_styles = debug_styles.to(device, non_blocking=True)
                save_stycona_debug_samples(
                    images=debug_images,
                    transform=stycona_runtime.transform,
                    epoch=epoch,
                    save_dir=debug_dir,
                    max_samples=max(args.stycona_debug_max_samples, 1),
                    style_images=debug_styles,
                )
            else:
                debug_images, _ = debug_batch
                debug_images = debug_images.to(device, non_blocking=True)
                save_stycona_debug_samples(
                    images=debug_images,
                    transform=stycona_runtime.transform,
                    epoch=epoch,
                    save_dir=debug_dir,
                    max_samples=max(args.stycona_debug_max_samples, 1),
                )

        train_loss = run_train_epoch(
            model=model,
            dataloader=loaders.train,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=config.train.mixed_precision and device.type == "cuda",
            stycona_runtime=stycona_runtime,
            log_interval=config.train.log_interval,
        )
        source_metrics = evaluate(model, loaders.val_source, device)
        if loaders.val_target is loaders.val_source:
            target_metrics = source_metrics
        else:
            target_metrics = evaluate(model, loaders.val_target, device)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} "
            f"val_dice={source_metrics['dice']:.4f} "
            f"target_val_dice={target_metrics['dice']:.4f}"
        )

        if target_metrics["dice"] > best_target_dice:
            best_target_dice = target_metrics["dice"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_target_dice": best_target_dice,
                    "config": asdict(config),
                },
                checkpoint_path,
            )
            print(f"Saved best checkpoint to {checkpoint_path} (target_dice={best_target_dice:.4f})")

    print(f"Training finished. Best target dice: {best_target_dice:.4f}")


if __name__ == "__main__":
    main()
