from dataclasses import dataclass, field, fields
from itertools import product
from typing import Any, Literal


Device = Literal["auto", "cpu", "cuda"]
ProgressBackend = Literal["auto", "notebook", "terminal"]
NoiseSchedule = Literal["linear", "cosine"]
ModelKind = Literal["vae", "gan", "diffusion"]


@dataclass(frozen=True)
class DataFixedConfig:
    data_dir: str = "data/cats"
    cache_dir: str = ".cache/fid"
    output_dir: str = "reports/runs"
    image_size: int = 64
    channels: int = 3
    pin_memory: bool = False
    num_workers: int = 0


@dataclass(frozen=True)
class DataGridConfig:
    train_fraction: float | list[float] = 0.9
    validation_fraction: float | list[float] = 0.1
    augment_flip: bool | list[bool] = True
    seed: int | list[int] = 42


@dataclass(frozen=True)
class TrainFixedConfig:
    device: Device = "auto"
    use_tqdm: bool = True
    progress_backend: ProgressBackend = "terminal"
    verbose: bool = True
    sample_grid_size: int = 64
    fid_real_samples: int = 1000
    fid_fake_samples: int = 1000
    early_stopping: bool = True
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0
    early_stopping_restore_best: bool = True


@dataclass(frozen=True)
class TrainGridConfig:
    epochs: int | list[int] = 30
    batch_size: int | list[int] = 64
    sample_every: int | list[int] = 5


@dataclass(frozen=True)
class VAEFixedConfig:
    base_channels: int = 32
    hidden_dim: int = 512


@dataclass(frozen=True)
class VAEGridConfig:
    latent_dim: int | list[int] = 128
    beta: float | list[float] = 1.0
    learning_rate: float | list[float] = 1e-3
    weight_decay: float | list[float] = 0.0


@dataclass(frozen=True)
class GANFixedConfig:
    spectral_norm: bool = True
    mode_collapse_window: int = 50
    mode_collapse_min_std: float = 1e-3


@dataclass(frozen=True)
class GANGridConfig:
    latent_dim: int | list[int] = 128
    generator_features: int | list[int] = 64
    discriminator_features: int | list[int] = 64
    learning_rate_g: float | list[float] = 2e-4
    learning_rate_d: float | list[float] = 2e-4
    label_smoothing: float | list[float] = 0.1
    n_disc_steps: int | list[int] = 1
    beta1: float | list[float] = 0.5
    beta2: float | list[float] = 0.999


@dataclass(frozen=True)
class DiffusionFixedConfig:
    time_embedding_dim: int = 128


@dataclass(frozen=True)
class DiffusionGridConfig:
    timesteps: int | list[int] = 400
    schedule: NoiseSchedule | list[NoiseSchedule] = "cosine"
    base_channels: int | list[int] = 64
    channel_mults: tuple[int, ...] | list[tuple[int, ...]] = (1, 2, 2, 2)
    learning_rate: float | list[float] = 2e-4
    beta_start: float | list[float] = 1e-4
    beta_end: float | list[float] = 0.02
    ema_decay: float | list[float] = 0.999


@dataclass(frozen=True)
class VAEExperiment:
    name: str
    data_fixed: DataFixedConfig = field(default_factory=DataFixedConfig)
    data_grid: DataGridConfig = field(default_factory=DataGridConfig)
    train_fixed: TrainFixedConfig = field(default_factory=TrainFixedConfig)
    train_grid: TrainGridConfig = field(default_factory=TrainGridConfig)
    model_fixed: VAEFixedConfig = field(default_factory=VAEFixedConfig)
    model_grid: VAEGridConfig = field(default_factory=VAEGridConfig)
    kind: ModelKind = "vae"


@dataclass(frozen=True)
class GANExperiment:
    name: str
    data_fixed: DataFixedConfig = field(default_factory=DataFixedConfig)
    data_grid: DataGridConfig = field(default_factory=DataGridConfig)
    train_fixed: TrainFixedConfig = field(default_factory=TrainFixedConfig)
    train_grid: TrainGridConfig = field(default_factory=TrainGridConfig)
    model_fixed: GANFixedConfig = field(default_factory=GANFixedConfig)
    model_grid: GANGridConfig = field(default_factory=GANGridConfig)
    kind: ModelKind = "gan"


@dataclass(frozen=True)
class DiffusionExperiment:
    name: str
    data_fixed: DataFixedConfig = field(default_factory=DataFixedConfig)
    data_grid: DataGridConfig = field(default_factory=DataGridConfig)
    train_fixed: TrainFixedConfig = field(default_factory=TrainFixedConfig)
    train_grid: TrainGridConfig = field(default_factory=TrainGridConfig)
    model_fixed: DiffusionFixedConfig = field(default_factory=DiffusionFixedConfig)
    model_grid: DiffusionGridConfig = field(default_factory=DiffusionGridConfig)
    kind: ModelKind = "diffusion"


Experiment = VAEExperiment | GANExperiment | DiffusionExperiment


def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def expand_grid(grid_instance: Any) -> list[dict[str, Any]]:
    keys = [field_.name for field_ in fields(grid_instance)]
    values = [as_list(getattr(grid_instance, key)) for key in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def expand_experiment_grid(experiment: Experiment) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for data_params, train_params, model_params in product(
        expand_grid(experiment.data_grid),
        expand_grid(experiment.train_grid),
        expand_grid(experiment.model_grid),
    ):
        runs.append(
            {
                "experiment": experiment.name,
                "kind": experiment.kind,
                "data": data_params,
                "train": train_params,
                "model": model_params,
            }
        )
    return runs


def experiment_grid_dataframe(experiment: Experiment):
    import pandas as pd

    rows = []
    for run in expand_experiment_grid(experiment):
        rows.append(
            {
                "experiment": run["experiment"],
                "kind": run["kind"],
                **{f"data.{key}": value for key, value in run["data"].items()},
                **{f"train.{key}": value for key, value in run["train"].items()},
                **{f"model.{key}": value for key, value in run["model"].items()},
            }
        )
    return pd.DataFrame(rows)
