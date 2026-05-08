from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    """
    Hai kiểu bố cục:
    - nested: {root}/{source_domain}/{train_split|val_split}, target val tại {root}/{target_domain}/{val_split}
    - flat: {root}/{train_split}, {root}/{val_split}, target metric có thể trùng val hoặc folder khác (target_val_split).
    """

    root_dir: str
    image_size: int
    num_workers: int
    layout: str
    source_domain: str
    target_domain: str
    train_split: str
    val_split: str
    test_split: str
    target_val_split: str | None


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    seed: int
    mixed_precision: bool
    log_interval: int


@dataclass(frozen=True)
class StyConaConfig:
    enable: bool
    prob: float
    min_step: int
    min_start: int
    k_vectors: int
    # paired_styles: auxiliary = file {id}_styleK_img.png cùng id; batch_shuffle: auxiliary = ảnh khác trong batch
    auxiliary_source: str
    # random | first | cycle — chỉ dùng khi auxiliary_source == paired_styles
    style_variant_sampling: str


@dataclass(frozen=True)
class RuntimeConfig:
    output_dir: str
    checkpoint_name: str
    device: str


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig
    train: TrainConfig
    stycona: StyConaConfig
    runtime: RuntimeConfig


def _read_yaml(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid yaml structure in {config_path}")
    return data


def _require_keys(payload: dict[str, Any], required_keys: tuple[str, ...], config_path: Path) -> None:
    missing = [key for key in required_keys if key not in payload]
    if missing:
        missing_str = ", ".join(missing)
        raise KeyError(f"Missing keys in {config_path}: {missing_str}")


def _data_config_from_dict(data: dict[str, Any]) -> DataConfig:
    layout = data.get("layout", "nested")
    if layout not in ("nested", "flat"):
        raise ValueError(f'data.layout must be "nested" or "flat", got {layout!r}')
    return DataConfig(
        root_dir=data["root_dir"],
        image_size=data["image_size"],
        num_workers=data["num_workers"],
        layout=layout,
        source_domain=data.get("source_domain", "original"),
        target_domain=data.get("target_domain", "augmented"),
        train_split=data.get("train_split", "train"),
        val_split=data.get("val_split", "val"),
        test_split=data.get("test_split", "test"),
        target_val_split=data.get("target_val_split"),
    )


def _stycona_config_from_dict(data: dict[str, Any]) -> StyConaConfig:
    aux = data.get("auxiliary_source", "paired_styles")
    if aux not in ("paired_styles", "batch_shuffle"):
        raise ValueError(f'stycona.auxiliary_source must be "paired_styles" or "batch_shuffle", got {aux!r}')
    sampling = data.get("style_variant_sampling", "random")
    if sampling not in ("random", "first", "cycle"):
        raise ValueError(f'stycona.style_variant_sampling must be "random", "first", or "cycle", got {sampling!r}')
    return StyConaConfig(
        enable=data["enable"],
        prob=data["prob"],
        min_step=data["min_step"],
        min_start=data["min_start"],
        k_vectors=data["k_vectors"],
        auxiliary_source=aux,
        style_variant_sampling=sampling,
    )


def load_config(config_path: str) -> ExperimentConfig:
    path = Path(config_path)
    payload = _read_yaml(path)
    _require_keys(payload, ("data", "train", "stycona", "runtime"), path)

    data_payload = payload["data"]
    if not isinstance(data_payload, dict):
        raise ValueError(f'"data" must be a mapping in {path}')
    stycona_payload = payload["stycona"]
    if not isinstance(stycona_payload, dict):
        raise ValueError(f'"stycona" must be a mapping in {path}')

    return ExperimentConfig(
        data=_data_config_from_dict(data_payload),
        train=TrainConfig(**payload["train"]),
        stycona=_stycona_config_from_dict(stycona_payload),
        runtime=RuntimeConfig(**payload["runtime"]),
    )
