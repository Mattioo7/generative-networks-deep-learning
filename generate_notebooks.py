"""Generate the three project notebooks from a shared cell template.

Run once after editing this file:
    python generate_notebooks.py
The three .ipynb files are overwritten in place.
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_METADATA = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.12",
    },
}


def code_cell(source: str) -> dict:
    lines = source.splitlines(keepends=True)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


def markdown_cell(source: str) -> dict:
    lines = source.splitlines(keepends=True)
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines,
    }


# ---------------------------------------------------------------------------
# VAE notebook
# ---------------------------------------------------------------------------

VAE_IMPORTS = '''\
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from scripts import (
    DataFixedConfig,
    DataGridConfig,
    TrainFixedConfig,
    TrainGridConfig,
    VAEExperiment,
    VAEFixedConfig,
    VAEGridConfig,
    build_dataloaders,
    build_vae,
    compute_fid,
    count_parameters,
    expand_experiment_grid,
    interpolate_vae,
    output_paths,
    prepare_data,
    report_fid,
    run_name,
    sample_vae_latents,
    save_history,
    save_history_plot,
    save_interpolation_grid,
    save_json,
    save_sample_grid,
    train_vae,
    vae_sample_iterator,
)
'''

VAE_PARAMETERS = '''\
# Training time on a free Colab T4 GPU at image_size=64 and ~14k cats:
#   batch_size=64, latent_dim=128, base_channels=32 -> ~30s per epoch.
#   30 epochs ~= 15 minutes total. Each beta sweep multiplies that by len(beta).

EXPERIMENT_NAME = "vae"

# --- Data ---------------------------------------------------------------
DATA_DIR = "data/cats"           # local path; on Kaggle use "/kaggle/input/cat-dataset/cats"
OUTPUT_DIR = "reports/runs"
CACHE_DIR = ".cache/fid"
IMAGE_SIZE = 64
TRAIN_FRACTION = 0.9
VALIDATION_FRACTION = 0.1
AUGMENT_FLIP = True

# --- Model --------------------------------------------------------------
LATENT_DIM = 128
HIDDEN_DIM = 512
BASE_CHANNELS = 32
BETA = 1.0

# --- Optimization -------------------------------------------------------
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
EPOCHS = 30
BATCH_SIZE = 64
SEED = 42

# --- Logging / sampling -------------------------------------------------
SAMPLE_EVERY = 5
FID_REAL_SAMPLES = 1000
FID_FAKE_SAMPLES = 1000
DEVICE = "auto"            # "auto" | "cpu" | "cuda"
PROGRESS_BACKEND = "auto"  # "auto" | "notebook" | "terminal"

experiment = VAEExperiment(
    name=EXPERIMENT_NAME,
    data_fixed=DataFixedConfig(
        data_dir=DATA_DIR,
        cache_dir=CACHE_DIR,
        output_dir=OUTPUT_DIR,
        image_size=IMAGE_SIZE,
    ),
    data_grid=DataGridConfig(
        train_fraction=TRAIN_FRACTION,
        validation_fraction=VALIDATION_FRACTION,
        augment_flip=AUGMENT_FLIP,
        seed=SEED,
    ),
    train_fixed=TrainFixedConfig(
        device=DEVICE,
        progress_backend=PROGRESS_BACKEND,
        fid_real_samples=FID_REAL_SAMPLES,
        fid_fake_samples=FID_FAKE_SAMPLES,
    ),
    train_grid=TrainGridConfig(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        sample_every=SAMPLE_EVERY,
    ),
    model_fixed=VAEFixedConfig(
        base_channels=BASE_CHANNELS,
        hidden_dim=HIDDEN_DIM,
    ),
    model_grid=VAEGridConfig(
        latent_dim=LATENT_DIM,
        beta=BETA,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    ),
)

runs = expand_experiment_grid(experiment)
run = runs[0]
print(f"Generated {len(runs)} configuration(s); the first is selected for this notebook run.")
'''

VAE_SETUP = '''\
torch.manual_seed(run["data"]["seed"])
np.random.seed(run["data"]["seed"])

run_dir = run_name(experiment.kind, run)
paths = output_paths(experiment.name, experiment.data_fixed.output_dir, run_dir)
save_json(paths["configs"] / "run.json", {"experiment": experiment.name, **run})
print(f"Run output: {paths['root']}")

prepared = prepare_data(
    experiment.data_fixed,
    train_fraction=run["data"]["train_fraction"],
    validation_fraction=run["data"]["validation_fraction"],
    augment_flip=run["data"]["augment_flip"],
    seed=run["data"]["seed"],
)
print(f"Train images: {len(prepared.train_dataset)} | Validation: {len(prepared.validation_dataset)}")

train_loader, validation_loader = build_dataloaders(
    prepared,
    experiment.data_fixed,
    batch_size=run["train"]["batch_size"],
    seed=run["data"]["seed"],
)

model = build_vae(run["model"], experiment.model_fixed, experiment.data_fixed)
print(f"VAE parameters: {count_parameters(model):,}")
'''

VAE_TRAINING = '''\
training_result = train_vae(
    model,
    train_loader,
    validation_loader,
    model_params=run["model"],
    train_params=run["train"],
    train_fixed=experiment.train_fixed,
    sample_dir=paths["samples"],
)

history = training_result["history"]
save_history(history, paths["metrics"] / "history.csv")
save_history_plot(
    history,
    paths["figures"] / "loss_curves.png",
    metrics=["loss", "validation_loss", "reconstruction", "kl"],
    title="VAE Loss Curves",
)
print(f"Trained in {training_result['elapsed_seconds']:.1f}s on {training_result['device']}.")
'''

VAE_EVALUATION = '''\
device = next(model.parameters()).device

with torch.no_grad():
    z = torch.randn(64, model.latent_dim, device=device)
    save_sample_grid(model.decode(z), paths["figures"] / "samples_final.png", nrow=8)

metrics_df = pd.DataFrame(history)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(metrics_df["epoch"], metrics_df["loss"], label="train")
axes[0].plot(metrics_df["epoch"], metrics_df["validation_loss"], label="validation")
axes[0].set_title("Total loss")
axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(metrics_df["epoch"], metrics_df["reconstruction"], label="reconstruction")
axes[1].plot(metrics_df["epoch"], metrics_df["kl"], label="KL")
axes[1].set_title("Loss components")
axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()

sampler = vae_sample_iterator(
    model,
    num_samples=experiment.train_fixed.fid_fake_samples,
    batch_size=run["train"]["batch_size"],
    device=device,
    seed=run["data"]["seed"],
)
fid_start = time.perf_counter()
fid_score = compute_fid(
    sampler,
    experiment.data_fixed,
    num_real_samples=experiment.train_fixed.fid_real_samples,
    seed=run["data"]["seed"],
    device=device,
)
fid_elapsed = time.perf_counter() - fid_start
report_fid(fid_score, elapsed_seconds=fid_elapsed)
save_json(paths["metrics"] / "fid.json", {"fid": fid_score, "elapsed_seconds": fid_elapsed})

img = plt.imread(paths["figures"] / "samples_final.png")
plt.figure(figsize=(8, 8))
plt.imshow(img); plt.axis("off")
plt.title(f"VAE samples (FID={fid_score:.2f})")
plt.show()
'''

VAE_INTERPOLATION = '''\
z1, z2 = sample_vae_latents(model, device, run["data"]["seed"])
images = interpolate_vae(model, z1, z2, num_steps=10)
interpolation_path = paths["figures"] / "interpolation.png"
save_interpolation_grid(images, interpolation_path)

img = plt.imread(interpolation_path)
plt.figure(figsize=(20, 2.6))
plt.imshow(img); plt.axis("off")
plt.title("VAE latent interpolation: z1 -> z2 (10 steps)")
plt.show()
'''


# ---------------------------------------------------------------------------
# GAN notebook
# ---------------------------------------------------------------------------

GAN_IMPORTS = '''\
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from scripts import (
    DataFixedConfig,
    DataGridConfig,
    GANExperiment,
    GANFixedConfig,
    GANGridConfig,
    TrainFixedConfig,
    TrainGridConfig,
    build_dataloaders,
    build_gan,
    compute_fid,
    count_parameters,
    expand_experiment_grid,
    gan_sample_iterator,
    interpolate_gan,
    output_paths,
    prepare_data,
    report_fid,
    run_name,
    sample_gan_latents,
    save_history,
    save_history_plot,
    save_interpolation_grid,
    save_json,
    save_sample_grid,
    train_gan,
)
'''

GAN_PARAMETERS = '''\
# Training time on a free Colab T4 GPU at image_size=64 and ~14k cats:
#   batch_size=128, generator/discriminator features=64 -> ~30s per epoch.
#   50 epochs ~= 25 minutes total. Mode collapse, when it triggers, surfaces
#   in the per-epoch warning printed by train_gan.

EXPERIMENT_NAME = "gan"

# --- Data ---------------------------------------------------------------
DATA_DIR = "data/cats"
OUTPUT_DIR = "reports/runs"
CACHE_DIR = ".cache/fid"
IMAGE_SIZE = 64
TRAIN_FRACTION = 0.95
VALIDATION_FRACTION = 0.05
AUGMENT_FLIP = True

# --- Generator / Discriminator -----------------------------------------
LATENT_DIM = 128
GENERATOR_FEATURES = 64
DISCRIMINATOR_FEATURES = 64
SPECTRAL_NORM = True
LABEL_SMOOTHING = 0.1
N_DISC_STEPS = 1

# --- Optimization -------------------------------------------------------
LEARNING_RATE_G = 2e-4
LEARNING_RATE_D = 2e-4
ADAM_BETA1 = 0.5
ADAM_BETA2 = 0.999
EPOCHS = 50
BATCH_SIZE = 128
SEED = 42

# --- Mode collapse heuristic -------------------------------------------
MODE_COLLAPSE_WINDOW = 50
MODE_COLLAPSE_MIN_STD = 1e-3

# --- Logging / sampling -------------------------------------------------
SAMPLE_EVERY = 5
FID_REAL_SAMPLES = 1000
FID_FAKE_SAMPLES = 1000
DEVICE = "auto"
PROGRESS_BACKEND = "auto"

experiment = GANExperiment(
    name=EXPERIMENT_NAME,
    data_fixed=DataFixedConfig(
        data_dir=DATA_DIR,
        cache_dir=CACHE_DIR,
        output_dir=OUTPUT_DIR,
        image_size=IMAGE_SIZE,
    ),
    data_grid=DataGridConfig(
        train_fraction=TRAIN_FRACTION,
        validation_fraction=VALIDATION_FRACTION,
        augment_flip=AUGMENT_FLIP,
        seed=SEED,
    ),
    train_fixed=TrainFixedConfig(
        device=DEVICE,
        progress_backend=PROGRESS_BACKEND,
        fid_real_samples=FID_REAL_SAMPLES,
        fid_fake_samples=FID_FAKE_SAMPLES,
    ),
    train_grid=TrainGridConfig(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        sample_every=SAMPLE_EVERY,
    ),
    model_fixed=GANFixedConfig(
        spectral_norm=SPECTRAL_NORM,
        mode_collapse_window=MODE_COLLAPSE_WINDOW,
        mode_collapse_min_std=MODE_COLLAPSE_MIN_STD,
    ),
    model_grid=GANGridConfig(
        latent_dim=LATENT_DIM,
        generator_features=GENERATOR_FEATURES,
        discriminator_features=DISCRIMINATOR_FEATURES,
        learning_rate_g=LEARNING_RATE_G,
        learning_rate_d=LEARNING_RATE_D,
        label_smoothing=LABEL_SMOOTHING,
        n_disc_steps=N_DISC_STEPS,
        beta1=ADAM_BETA1,
        beta2=ADAM_BETA2,
    ),
)

runs = expand_experiment_grid(experiment)
run = runs[0]
print(f"Generated {len(runs)} configuration(s); the first is selected for this notebook run.")
'''

GAN_SETUP = '''\
torch.manual_seed(run["data"]["seed"])
np.random.seed(run["data"]["seed"])

run_dir = run_name(experiment.kind, run)
paths = output_paths(experiment.name, experiment.data_fixed.output_dir, run_dir)
save_json(paths["configs"] / "run.json", {"experiment": experiment.name, **run})
print(f"Run output: {paths['root']}")

prepared = prepare_data(
    experiment.data_fixed,
    train_fraction=run["data"]["train_fraction"],
    validation_fraction=run["data"]["validation_fraction"],
    augment_flip=run["data"]["augment_flip"],
    seed=run["data"]["seed"],
)
print(f"Train images: {len(prepared.train_dataset)} | Validation: {len(prepared.validation_dataset)}")

train_loader, _ = build_dataloaders(
    prepared,
    experiment.data_fixed,
    batch_size=run["train"]["batch_size"],
    seed=run["data"]["seed"],
)

generator, discriminator = build_gan(run["model"], experiment.model_fixed, experiment.data_fixed)
print(
    f"Generator parameters: {count_parameters(generator):,} | "
    f"Discriminator parameters: {count_parameters(discriminator):,}"
)
'''

GAN_TRAINING = '''\
training_result = train_gan(
    generator,
    discriminator,
    train_loader,
    model_params=run["model"],
    train_params=run["train"],
    train_fixed=experiment.train_fixed,
    fixed_params=experiment.model_fixed,
    sample_dir=paths["samples"],
)

history = training_result["history"]
collapse_events = training_result["mode_collapse_events"]
save_history(history, paths["metrics"] / "history.csv")
save_history_plot(
    history,
    paths["figures"] / "loss_curves.png",
    metrics=["d_loss", "g_loss", "real_logit", "fake_logit"],
    title="GAN Loss Curves",
)
save_json(paths["metrics"] / "mode_collapse.json", {"events": collapse_events})
print(
    f"Trained in {training_result['elapsed_seconds']:.1f}s on {training_result['device']}; "
    f"mode-collapse warnings: {len(collapse_events)}."
)
'''

GAN_EVALUATION = '''\
device = next(generator.parameters()).device

with torch.no_grad():
    noise = torch.randn(64, generator.latent_dim, 1, 1, device=device)
    save_sample_grid(generator(noise), paths["figures"] / "samples_final.png", nrow=8)

metrics_df = pd.DataFrame(history)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(metrics_df["epoch"], metrics_df["d_loss"], label="discriminator")
axes[0].plot(metrics_df["epoch"], metrics_df["g_loss"], label="generator")
axes[0].set_title("Adversarial losses")
axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(metrics_df["epoch"], metrics_df["real_logit"], label="D(real)")
axes[1].plot(metrics_df["epoch"], metrics_df["fake_logit"], label="D(fake)")
axes[1].set_title("Discriminator output mean")
axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()

sampler = gan_sample_iterator(
    generator,
    num_samples=experiment.train_fixed.fid_fake_samples,
    batch_size=run["train"]["batch_size"],
    device=device,
    seed=run["data"]["seed"],
)
fid_start = time.perf_counter()
fid_score = compute_fid(
    sampler,
    experiment.data_fixed,
    num_real_samples=experiment.train_fixed.fid_real_samples,
    seed=run["data"]["seed"],
    device=device,
)
fid_elapsed = time.perf_counter() - fid_start
report_fid(fid_score, elapsed_seconds=fid_elapsed)
save_json(paths["metrics"] / "fid.json", {"fid": fid_score, "elapsed_seconds": fid_elapsed})

img = plt.imread(paths["figures"] / "samples_final.png")
plt.figure(figsize=(8, 8))
plt.imshow(img); plt.axis("off")
plt.title(f"GAN samples (FID={fid_score:.2f})")
plt.show()
'''

GAN_INTERPOLATION = '''\
z1, z2 = sample_gan_latents(generator, device, run["data"]["seed"])
images = interpolate_gan(generator, z1, z2, num_steps=10)
interpolation_path = paths["figures"] / "interpolation.png"
save_interpolation_grid(images, interpolation_path)

img = plt.imread(interpolation_path)
plt.figure(figsize=(20, 2.6))
plt.imshow(img); plt.axis("off")
plt.title("GAN latent interpolation: z1 -> z2 (10 steps)")
plt.show()
'''


# ---------------------------------------------------------------------------
# Diffusion notebook
# ---------------------------------------------------------------------------

DIFF_IMPORTS = '''\
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from scripts import (
    DataFixedConfig,
    DataGridConfig,
    DiffusionExperiment,
    DiffusionFixedConfig,
    DiffusionGridConfig,
    TrainFixedConfig,
    TrainGridConfig,
    build_dataloaders,
    build_diffusion,
    compute_fid,
    count_parameters,
    diffusion_sample_iterator,
    expand_experiment_grid,
    interpolate_diffusion,
    output_paths,
    prepare_data,
    report_fid,
    run_name,
    sample_diffusion_latents,
    save_history,
    save_history_plot,
    save_interpolation_grid,
    save_json,
    save_sample_grid,
    train_diffusion,
)
'''

DIFF_PARAMETERS = '''\
# Training time on a free Colab T4 GPU at image_size=64 and ~14k cats:
#   batch_size=64, base_channels=64, T=400 timesteps -> ~75s per epoch.
#   30 epochs ~= 38 minutes training; another ~6 minutes per FID computation
#   because reverse sampling at T=400 dominates wall-clock time.
# Reduce T to 200 to halve sampling cost at modest quality loss.

EXPERIMENT_NAME = "diffusion"

# --- Data ---------------------------------------------------------------
DATA_DIR = "data/cats"
OUTPUT_DIR = "reports/runs"
CACHE_DIR = ".cache/fid"
IMAGE_SIZE = 64
TRAIN_FRACTION = 0.95
VALIDATION_FRACTION = 0.05
AUGMENT_FLIP = True

# --- UNet / scheduler ---------------------------------------------------
TIMESTEPS = 400                # 200..500 is the practical Colab range
SCHEDULE = "cosine"            # "linear" | "cosine"
BASE_CHANNELS = 64
CHANNEL_MULTS = (1, 2, 2, 2)   # 4 resolutions: 64 -> 32 -> 16 -> 8
TIME_EMBEDDING_DIM = 128
BETA_START = 1e-4
BETA_END = 0.02
EMA_DECAY = 0.999

# --- Optimization -------------------------------------------------------
LEARNING_RATE = 2e-4
EPOCHS = 30
BATCH_SIZE = 64
SEED = 42

# --- Logging / sampling -------------------------------------------------
SAMPLE_EVERY = 10              # diffusion sampling is expensive; log sparingly
FID_REAL_SAMPLES = 1000
FID_FAKE_SAMPLES = 1000
DEVICE = "auto"
PROGRESS_BACKEND = "auto"

experiment = DiffusionExperiment(
    name=EXPERIMENT_NAME,
    data_fixed=DataFixedConfig(
        data_dir=DATA_DIR,
        cache_dir=CACHE_DIR,
        output_dir=OUTPUT_DIR,
        image_size=IMAGE_SIZE,
    ),
    data_grid=DataGridConfig(
        train_fraction=TRAIN_FRACTION,
        validation_fraction=VALIDATION_FRACTION,
        augment_flip=AUGMENT_FLIP,
        seed=SEED,
    ),
    train_fixed=TrainFixedConfig(
        device=DEVICE,
        progress_backend=PROGRESS_BACKEND,
        fid_real_samples=FID_REAL_SAMPLES,
        fid_fake_samples=FID_FAKE_SAMPLES,
    ),
    train_grid=TrainGridConfig(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        sample_every=SAMPLE_EVERY,
    ),
    model_fixed=DiffusionFixedConfig(
        time_embedding_dim=TIME_EMBEDDING_DIM,
    ),
    model_grid=DiffusionGridConfig(
        timesteps=TIMESTEPS,
        schedule=SCHEDULE,
        base_channels=BASE_CHANNELS,
        channel_mults=CHANNEL_MULTS,
        learning_rate=LEARNING_RATE,
        beta_start=BETA_START,
        beta_end=BETA_END,
        ema_decay=EMA_DECAY,
    ),
)

runs = expand_experiment_grid(experiment)
run = runs[0]
print(f"Generated {len(runs)} configuration(s); the first is selected for this notebook run.")
'''

DIFF_SETUP = '''\
torch.manual_seed(run["data"]["seed"])
np.random.seed(run["data"]["seed"])

run_dir = run_name(experiment.kind, run)
paths = output_paths(experiment.name, experiment.data_fixed.output_dir, run_dir)
save_json(paths["configs"] / "run.json", {"experiment": experiment.name, **run})
print(f"Run output: {paths['root']}")

prepared = prepare_data(
    experiment.data_fixed,
    train_fraction=run["data"]["train_fraction"],
    validation_fraction=run["data"]["validation_fraction"],
    augment_flip=run["data"]["augment_flip"],
    seed=run["data"]["seed"],
)
print(f"Train images: {len(prepared.train_dataset)} | Validation: {len(prepared.validation_dataset)}")

train_loader, validation_loader = build_dataloaders(
    prepared,
    experiment.data_fixed,
    batch_size=run["train"]["batch_size"],
    seed=run["data"]["seed"],
)

model = build_diffusion(run["model"], experiment.model_fixed, experiment.data_fixed)
print(f"Diffusion parameters: {count_parameters(model):,}")
'''

DIFF_TRAINING = '''\
training_result = train_diffusion(
    model,
    train_loader,
    validation_loader,
    model_params=run["model"],
    train_params=run["train"],
    train_fixed=experiment.train_fixed,
    sample_dir=paths["samples"],
)

history = training_result["history"]
save_history(history, paths["metrics"] / "history.csv")
save_history_plot(
    history,
    paths["figures"] / "loss_curves.png",
    metrics=["loss", "validation_loss"],
    title="Diffusion Loss Curves",
)
print(f"Trained in {training_result['elapsed_seconds']:.1f}s on {training_result['device']}.")
'''

DIFF_EVALUATION = '''\
device = next(model.parameters()).device

with torch.no_grad():
    samples = model.sample(16, device)
    save_sample_grid(samples, paths["figures"] / "samples_final.png", nrow=4)

metrics_df = pd.DataFrame(history)
fig, axis = plt.subplots(figsize=(8, 5))
axis.plot(metrics_df["epoch"], metrics_df["loss"], label="train")
axis.plot(metrics_df["epoch"], metrics_df["validation_loss"], label="validation")
axis.set_title("Diffusion training loss"); axis.set_xlabel("Epoch")
axis.legend(); axis.grid(alpha=0.3)
plt.tight_layout(); plt.show()

sampler = diffusion_sample_iterator(
    model,
    num_samples=experiment.train_fixed.fid_fake_samples,
    batch_size=run["train"]["batch_size"],
    device=device,
    seed=run["data"]["seed"],
)
fid_start = time.perf_counter()
fid_score = compute_fid(
    sampler,
    experiment.data_fixed,
    num_real_samples=experiment.train_fixed.fid_real_samples,
    seed=run["data"]["seed"],
    device=device,
)
fid_elapsed = time.perf_counter() - fid_start
report_fid(fid_score, elapsed_seconds=fid_elapsed)
save_json(paths["metrics"] / "fid.json", {"fid": fid_score, "elapsed_seconds": fid_elapsed})

img = plt.imread(paths["figures"] / "samples_final.png")
plt.figure(figsize=(8, 8))
plt.imshow(img); plt.axis("off")
plt.title(f"Diffusion samples (FID={fid_score:.2f})")
plt.show()
'''

DIFF_INTERPOLATION = '''\
z1, z2 = sample_diffusion_latents(model, device, run["data"]["seed"])
images = interpolate_diffusion(model, z1, z2, num_steps=10)
interpolation_path = paths["figures"] / "interpolation.png"
save_interpolation_grid(images, interpolation_path)

img = plt.imread(interpolation_path)
plt.figure(figsize=(20, 2.6))
plt.imshow(img); plt.axis("off")
plt.title("Diffusion latent interpolation: noise z1 -> noise z2 (10 steps)")
plt.show()
'''


# ---------------------------------------------------------------------------
# Notebook builders
# ---------------------------------------------------------------------------


def build_notebook(title: str, sections: list[tuple[str, str]]) -> dict:
    cells: list[dict] = [markdown_cell(f"# {title}")]
    for header, source in sections:
        cells.append(markdown_cell(f"## {header}"))
        cells.append(code_cell(source))
    return {
        "cells": cells,
        "metadata": NOTEBOOK_METADATA,
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(path: Path, notebook: dict) -> None:
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    here = Path(__file__).resolve().parent
    write_notebook(
        here / "01_vae.ipynb",
        build_notebook(
            "Variational Autoencoder for cat images",
            [
                ("Imports", VAE_IMPORTS),
                ("Parameters", VAE_PARAMETERS),
                ("Setup", VAE_SETUP),
                ("Training", VAE_TRAINING),
                ("Evaluation", VAE_EVALUATION),
                ("Interpolation", VAE_INTERPOLATION),
            ],
        ),
    )
    write_notebook(
        here / "02_gan.ipynb",
        build_notebook(
            "DCGAN for cat images",
            [
                ("Imports", GAN_IMPORTS),
                ("Parameters", GAN_PARAMETERS),
                ("Setup", GAN_SETUP),
                ("Training", GAN_TRAINING),
                ("Evaluation", GAN_EVALUATION),
                ("Interpolation", GAN_INTERPOLATION),
            ],
        ),
    )
    write_notebook(
        here / "03_diffusion.ipynb",
        build_notebook(
            "DDPM diffusion model for cat images",
            [
                ("Imports", DIFF_IMPORTS),
                ("Parameters", DIFF_PARAMETERS),
                ("Setup", DIFF_SETUP),
                ("Training", DIFF_TRAINING),
                ("Evaluation", DIFF_EVALUATION),
                ("Interpolation", DIFF_INTERPOLATION),
            ],
        ),
    )
    print("Notebooks generated.")


if __name__ == "__main__":
    main()
