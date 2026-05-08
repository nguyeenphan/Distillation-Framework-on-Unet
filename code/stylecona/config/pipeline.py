from pathlib import Path

from .decompose import ExperimentConfig, load_config

DEFAULT_CONFIG_FILE = "default.yaml"


def load_experiment_config(config_path: str | None = None) -> ExperimentConfig:
    if config_path is None:
        base_dir = Path(__file__).resolve().parent
        config_path = str(base_dir / DEFAULT_CONFIG_FILE)
    return load_config(config_path)
