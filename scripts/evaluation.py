import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import sqrtm
from torch import nn
from torchvision.models import Inception_V3_Weights, inception_v3

from .config import DataFixedConfig
from .data import real_image_loader
from .models import DiffusionModel, Generator, VAE
from .progress import progress_bar, stage


INCEPTION_INPUT = 299


# ---------------------------------------------------------------------------
# InceptionV3 feature extractor (fc replaced with identity, pool features)
# ---------------------------------------------------------------------------


def load_inception(device: torch.device) -> nn.Module:
    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights, transform_input=False, aux_logits=True)
    model.fc = nn.Identity()
    model.eval()
    model.to(device)
    return model


def to_inception_tensor(images: torch.Tensor) -> torch.Tensor:
    images = images.clamp(-1.0, 1.0)
    images = (images + 1.0) / 2.0
    if images.size(-1) != INCEPTION_INPUT:
        images = nn.functional.interpolate(images, size=(INCEPTION_INPUT, INCEPTION_INPUT), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


@torch.no_grad()
def inception_features(
    images_iter: Iterable[torch.Tensor],
    inception: nn.Module,
    device: torch.device,
    *,
    use_tqdm: bool,
    description: str,
) -> np.ndarray:
    activations: list[np.ndarray] = []
    iterator = progress_bar(
        images_iter,
        enabled=use_tqdm,
        backend="terminal",
        description=description,
        leave=True,
    )
    for batch in iterator:
        batch = batch.to(device)
        prepared = to_inception_tensor(batch)
        features = inception(prepared)
        activations.append(features.detach().cpu().numpy())
    return np.concatenate(activations, axis=0)


def calculate_statistics(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = features.mean(axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


# ---------------------------------------------------------------------------
# Real-image FID statistics (cached on disk to avoid recomputing across runs)
# ---------------------------------------------------------------------------


def fid_cache_key(data_fixed: DataFixedConfig, num_samples: int, seed: int) -> str:
    payload = {
        "data_dir": str(Path(data_fixed.data_dir).resolve()),
        "image_size": data_fixed.image_size,
        "num_samples": num_samples,
        "seed": seed,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return digest


def real_fid_statistics(
    data_fixed: DataFixedConfig,
    num_samples: int,
    seed: int,
    device: torch.device,
    batch_size: int = 32,
    *,
    use_tqdm: bool = True,
    cache_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    cache_root = Path(cache_dir if cache_dir is not None else data_fixed.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    key = fid_cache_key(data_fixed, num_samples, seed)
    cache_path = cache_root / f"real_stats_{key}.npz"

    if cache_path.exists():
        cached = np.load(cache_path)
        return cached["mu"], cached["sigma"]

    inception = load_inception(device)
    loader = real_image_loader(data_fixed, num_samples, batch_size, seed)
    features = inception_features(
        loader,
        inception,
        device,
        use_tqdm=use_tqdm,
        description="Real images → Inception",
    )
    mu, sigma = calculate_statistics(features)
    np.savez(cache_path, mu=mu, sigma=sigma)
    return mu, sigma


# ---------------------------------------------------------------------------
# Generated-image iterators per model kind
# ---------------------------------------------------------------------------


@torch.no_grad()
def vae_sample_iterator(
    model: VAE,
    num_samples: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> Iterable[torch.Tensor]:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    remaining = num_samples
    while remaining > 0:
        size = min(batch_size, remaining)
        z = torch.randn(size, model.latent_dim, generator=generator, device=device)
        yield model.decode(z).detach()
        remaining -= size


@torch.no_grad()
def gan_sample_iterator(
    generator_model: Generator,
    num_samples: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> Iterable[torch.Tensor]:
    generator_model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    remaining = num_samples
    while remaining > 0:
        size = min(batch_size, remaining)
        noise = torch.randn(size, generator_model.latent_dim, 1, 1, generator=generator, device=device)
        yield generator_model(noise).detach()
        remaining -= size


@torch.no_grad()
def diffusion_sample_iterator(
    model: DiffusionModel,
    num_samples: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> Iterable[torch.Tensor]:
    model.eval()
    torch.manual_seed(seed)
    remaining = num_samples
    while remaining > 0:
        size = min(batch_size, remaining)
        yield model.sample(size, device).detach()
        remaining -= size


# ---------------------------------------------------------------------------
# Top-level FID
# ---------------------------------------------------------------------------


def compute_fid(
    sample_iterator: Iterable[torch.Tensor],
    data_fixed: DataFixedConfig,
    num_real_samples: int,
    seed: int,
    device: torch.device,
    *,
    use_tqdm: bool = True,
    real_batch_size: int = 32,
) -> float:
    mu_real, sigma_real = real_fid_statistics(
        data_fixed,
        num_samples=num_real_samples,
        seed=seed,
        device=device,
        batch_size=real_batch_size,
        use_tqdm=use_tqdm,
    )

    inception = load_inception(device)
    fake_features = inception_features(
        sample_iterator,
        inception,
        device,
        use_tqdm=use_tqdm,
        description="Generated → Inception",
    )
    mu_fake, sigma_fake = calculate_statistics(fake_features)
    return frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)


# ---------------------------------------------------------------------------
# Latent interpolation
# ---------------------------------------------------------------------------


def linear_interpolation(z1: torch.Tensor, z2: torch.Tensor, num_steps: int) -> torch.Tensor:
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2 to include endpoints.")
    alphas = torch.linspace(0.0, 1.0, num_steps, device=z1.device)
    weight_shape = (num_steps,) + (1,) * z1.dim()
    weights = alphas.view(weight_shape)
    z1_expanded = z1.unsqueeze(0)
    z2_expanded = z2.unsqueeze(0)
    return (1.0 - weights) * z1_expanded + weights * z2_expanded


@torch.no_grad()
def interpolate_vae(model: VAE, z1: torch.Tensor, z2: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
    model.eval()
    latents = linear_interpolation(z1, z2, num_steps)
    return model.decode(latents)


@torch.no_grad()
def interpolate_gan(generator: Generator, z1: torch.Tensor, z2: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
    generator.eval()
    latents = linear_interpolation(z1, z2, num_steps)
    if latents.dim() == 2:
        latents = latents.view(latents.size(0), latents.size(1), 1, 1)
    return generator(latents)


@torch.no_grad()
def interpolate_diffusion(
    model: DiffusionModel,
    z1: torch.Tensor,
    z2: torch.Tensor,
    num_steps: int = 10,
) -> torch.Tensor:
    model.eval()
    latents = linear_interpolation(z1, z2, num_steps)
    device = next(model.parameters()).device
    return model.sample(num_steps, device, initial_noise=latents)


def sample_vae_latents(model: VAE, device: torch.device, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    z1 = torch.randn(model.latent_dim, generator=generator, device=device)
    z2 = torch.randn(model.latent_dim, generator=generator, device=device)
    return z1, z2


def sample_gan_latents(generator: Generator, device: torch.device, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = torch.Generator(device=device).manual_seed(seed)
    z1 = torch.randn(generator.latent_dim, generator=rng, device=device)
    z2 = torch.randn(generator.latent_dim, generator=rng, device=device)
    return z1, z2


def sample_diffusion_latents(model: DiffusionModel, device: torch.device, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = torch.Generator(device=device).manual_seed(seed)
    shape = (model.channels, model.image_size, model.image_size)
    z1 = torch.randn(shape, generator=rng, device=device)
    z2 = torch.randn(shape, generator=rng, device=device)
    return z1, z2


def report_fid(
    fid: float,
    *,
    elapsed_seconds: float | None = None,
    enabled: bool = True,
) -> None:
    if not enabled:
        return
    suffix = f" (sampling took {int(elapsed_seconds // 3600):02d}:{int(elapsed_seconds % 3600 // 60):02d}:{int(elapsed_seconds % 60):02d})" if elapsed_seconds is not None else ""
    stage(f"FID: {fid:.3f}{suffix}", enabled=True)
