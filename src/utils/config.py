"""Configuration loading utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str | Path = "configs/default.yaml") -> DictConfig:
    """Load YAML configuration file using OmegaConf.

    Parameters
    ----------
    config_path : str or Path
        Path to YAML config file.

    Returns
    -------
    cfg : DictConfig
        Configuration object with dot-access support.

    Raises
    ------
    FileNotFoundError
        If config file does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = OmegaConf.load(str(config_path))
    return cfg


def merge_configs(base: DictConfig, override: DictConfig) -> DictConfig:
    """Merge two configurations, with override taking precedence."""
    return OmegaConf.merge(base, override)


def config_to_dict(cfg: DictConfig) -> dict[str, Any]:
    """Convert OmegaConf config to plain Python dict."""
    return OmegaConf.to_container(cfg, resolve=True)
