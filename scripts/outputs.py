import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torchvision.utils import make_grid, save_image


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def run_name(kind: str, run: dict[str, Any]) -> str:
    model = run["model"]
    train = run["train"]

    dt = datetime.now().strftime("%Y%m%d_%H%M%S")

    if kind == "vae":
        token = f"vae_z{model['latent_dim']}_b{model['beta']}_lr{model['learning_rate']}_e{train['epochs']}"
    elif kind == "gan":
        token = (
            f"gan_z{model['latent_dim']}_g{model['generator_features']}_d{model['discriminator_features']}"
            f"_lr{model['learning_rate_g']}_ls{model['label_smoothing']}_e{train['epochs']}"
        )
    elif kind == "diffusion":
        token = (
            f"diff_T{model['timesteps']}_{model['schedule']}_c{model['base_channels']}"
            f"_lr{model['learning_rate']}_e{train['epochs']}"
        )
    else:
        raise ValueError(f"Unknown kind: {kind}")

    return f"{dt}_{slugify(token)}"


def output_paths(experiment_name: str, output_root: str, run_dir_name: str) -> dict[str, Path]:
    root = Path(output_root) / experiment_name / run_dir_name
    paths = {
        "root": root,
        "figures": root / "figures",
        "metrics": root / "metrics",
        "configs": root / "configs",
        "samples": root / "samples",
        "checkpoints": root / "checkpoints",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, default=str))


def save_history(history: list[dict[str, float]], path: Path) -> None:
    pd.DataFrame(history).to_csv(path, index=False)


def save_history_plot(
    history: list[dict[str, float]],
    path: Path,
    metrics: list[str],
    title: str = "Training history",
) -> None:
    if not history:
        return

    data = pd.DataFrame(history)
    available = [name for name in metrics if name in data.columns]
    if not available:
        return

    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for name in available:
        axis.plot(data["epoch"], data[name], label=name)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Value")
    axis.set_title(title)
    axis.legend()
    axis.grid(True, alpha=0.3)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def denormalize(images: torch.Tensor) -> torch.Tensor:
    return ((images.detach().cpu().clamp(-1.0, 1.0) + 1.0) / 2.0)


def save_sample_grid(images: torch.Tensor, path: Path, nrow: int = 8) -> None:
    grid = make_grid(denormalize(images), nrow=nrow, padding=2)
    save_image(grid, path)


def save_interpolation_grid(images: torch.Tensor, path: Path) -> None:
    save_image(make_grid(denormalize(images), nrow=images.size(0), padding=2), path)


def save_checkpoint(state: dict[str, Any], path: Path) -> None:
    torch.save(state, path)
