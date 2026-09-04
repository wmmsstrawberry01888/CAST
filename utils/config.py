"""
YAML-based configuration management for CAST experiments.

Hierarchical merge order: defaults.yaml -> dataset/* -> model/* -> experiment/*
Each level overrides the previous. CLI arguments override everything.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    name: str = "weather"
    num_vertices: int = 4
    num_features: int = 3
    points_per_hour: int = 12


@dataclass
class ModelConfig:
    V: int = 4
    F: int = 3
    T_h: int = 12
    T_p: int = 12
    d_h: int = 32
    device: str = "auto"
    # Diffusion
    N: int = 200
    sample_steps: int = 200
    epsilon_theta: str = "UGnet"
    beta_end: float = 0.02
    beta_schedule: str = "quad"
    sample_strategy: str = "ddpm"
    channel_multipliers: List[int] = field(default_factory=lambda: [1, 2])
    supports_len: int = 2


@dataclass
class TrainConfig:
    epochs: int = 50
    lr: float = 1e-4
    batch_size: int = 16
    weight_decay: float = 1e-5
    optimizer: str = "adam"
    early_stop_patience: int = 10
    seed: int = 2025
    # Balancing coefficient between L_pred and L_cdg in the composite objective
    # L = L_pred + lambda_cdg * L_cdg (see Training objective in the paper).
    lambda_cdg: float = 1.0
    use_amp: bool = False
    grad_clip: float = 0.0
    lr_scheduler: str = "none"  # none | cosine | plateau | step
    lr_step_size: int = 10
    lr_gamma: float = 0.5


@dataclass
class ExperimentConfig:
    name: str = "default"
    model: str = "cast"
    data: str = "weather"
    mode: str = "standard"  # standard | k_fold | select
    k_folds: int = 5
    use_diffusion: bool = False
    # CAST ablation flags
    use_gat: bool = True
    use_temporal: bool = True
    use_spatio: bool = True
    use_network: bool = True
    # Paths
    output_dir: str = "./output"
    log_dir: str = "./logs"
    # Tags for tracking
    tags: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. override wins on conflict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _dict_to_dataclass(cls: type, d: dict):
    """Convert a (possibly nested) dict to the given dataclass type."""
    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    for key, value in d.items():
        if key not in field_types:
            continue
        if isinstance(value, dict) and hasattr(field_types[key], '__dataclass_fields__'):
            kwargs[key] = _dict_to_dataclass(field_types[key], value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _load_yaml(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def load_config(config_path: str, cli_overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Load and merge config from a YAML file, optionally with CLI overrides.

    Parameters
    ----------
    config_path : str
        Path to the experiment YAML config file.
    cli_overrides : dict, optional
        CLI overrides as key-value pairs (e.g. {'model.T_h': 96}).

    Returns
    -------
    Config
        Fully merged configuration.
    """
    experiment_dict = _load_yaml(config_path)

    # Determine project root (2 levels up from this file)
    project_root = Path(__file__).resolve().parent.parent

    # Layer 1: defaults
    defaults_path = project_root / "configs" / "defaults.yaml"
    if defaults_path.exists():
        merged = _load_yaml(str(defaults_path))
    else:
        merged = {}

    # Layer 2: experiment config (overrides defaults)
    merged = _deep_merge(merged, experiment_dict)

    # Layer 3: CLI overrides
    if cli_overrides:
        merged = _deep_merge(merged, cli_overrides)

    # Build Config from merged dict
    config = Config(
        data=_dict_to_dataclass(DataConfig, merged.get('data', {})),
        model=_dict_to_dataclass(ModelConfig, merged.get('model', {})),
        train=_dict_to_dataclass(TrainConfig, merged.get('train', {})),
        experiment=_dict_to_dataclass(ExperimentConfig, merged.get('experiment', {})),
    )

    # Sync cross-references
    config.model.V = config.data.num_vertices
    config.model.F = config.data.num_features
    config.model.T_p = config.model.T_h

    return config


def save_config(config: Config, path: str) -> None:
    """Save a Config to YAML."""
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
