import math
from typing import Any

import torch
from torch import nn
from torch.nn.utils import spectral_norm

from .config import (
    DataFixedConfig,
    DiffusionFixedConfig,
    GANFixedConfig,
    NoiseSchedule,
    VAEFixedConfig,
)


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------


class VAEEncoder(nn.Module):
    def __init__(self, channels: int, base_channels: int, hidden_dim: int, latent_dim: int, image_size: int):
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.conv = nn.Sequential(
            nn.Conv2d(channels, c1, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c3, c4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        spatial = image_size // 16
        if spatial < 1:
            raise ValueError(f"image_size={image_size} too small for VAE encoder (needs >=16)")
        self.spatial = spatial
        self.flat_dim = c4 * spatial * spatial
        self.fc = nn.Sequential(
            nn.Linear(self.flat_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.conv(inputs).flatten(1)
        hidden = self.fc(features)
        return self.fc_mu(hidden), self.fc_logvar(hidden)


class VAEDecoder(nn.Module):
    def __init__(self, channels: int, base_channels: int, hidden_dim: int, latent_dim: int, image_size: int):
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        spatial = image_size // 16
        self.spatial = spatial
        self.start_channels = c4
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, c4 * spatial * spatial),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(c4, c3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(c3, c2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(c2, c1, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(c1, channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        hidden = self.fc(z).view(-1, self.start_channels, self.spatial, self.spatial)
        return self.deconv(hidden)


class VAE(nn.Module):
    def __init__(
        self,
        channels: int,
        base_channels: int,
        hidden_dim: int,
        latent_dim: int,
        image_size: int,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.channels = channels
        self.encoder = VAEEncoder(channels, base_channels, hidden_dim, latent_dim, image_size)
        self.decoder = VAEDecoder(channels, base_channels, hidden_dim, latent_dim, image_size)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def encode(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(inputs)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(inputs)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def build_vae(
    model_params: dict[str, Any],
    model_fixed: VAEFixedConfig,
    data_fixed: DataFixedConfig,
) -> VAE:
    return VAE(
        channels=data_fixed.channels,
        base_channels=model_fixed.base_channels,
        hidden_dim=model_fixed.hidden_dim,
        latent_dim=model_params["latent_dim"],
        image_size=data_fixed.image_size,
    )


# ---------------------------------------------------------------------------
# GAN (DCGAN with optional spectral normalization)
# ---------------------------------------------------------------------------


class Generator(nn.Module):
    def __init__(self, latent_dim: int, channels: int, features: int, image_size: int):
        super().__init__()
        if image_size != 64:
            raise NotImplementedError(
                "DCGAN generator is wired for 64x64 outputs; pass image_size=64."
            )
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, features * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(features * 8, features * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(features * 4, features * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(features * 2, features, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(features, channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        if noise.dim() == 2:
            noise = noise.view(noise.size(0), noise.size(1), 1, 1)
        return self.net(noise)


def _maybe_sn(module: nn.Module, enabled: bool) -> nn.Module:
    return spectral_norm(module) if enabled else module


class Discriminator(nn.Module):
    def __init__(self, channels: int, features: int, image_size: int, use_spectral_norm: bool):
        super().__init__()
        if image_size != 64:
            raise NotImplementedError(
                "DCGAN discriminator is wired for 64x64 inputs; pass image_size=64."
            )
        self.image_size = image_size
        self.net = nn.Sequential(
            _maybe_sn(nn.Conv2d(channels, features, kernel_size=4, stride=2, padding=1, bias=False), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            _maybe_sn(nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1, bias=False), use_spectral_norm),
            nn.BatchNorm2d(features * 2) if not use_spectral_norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
            _maybe_sn(nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1, bias=False), use_spectral_norm),
            nn.BatchNorm2d(features * 4) if not use_spectral_norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
            _maybe_sn(nn.Conv2d(features * 4, features * 8, kernel_size=4, stride=2, padding=1, bias=False), use_spectral_norm),
            nn.BatchNorm2d(features * 8) if not use_spectral_norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
            _maybe_sn(nn.Conv2d(features * 8, 1, kernel_size=4, stride=1, padding=0, bias=False), use_spectral_norm),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs).view(inputs.size(0))


def init_dcgan_weights(module: nn.Module) -> None:
    classname = module.__class__.__name__
    if "Conv" in classname:
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias.data, 0)


def build_gan(
    model_params: dict[str, Any],
    model_fixed: GANFixedConfig,
    data_fixed: DataFixedConfig,
) -> tuple[Generator, Discriminator]:
    generator = Generator(
        latent_dim=model_params["latent_dim"],
        channels=data_fixed.channels,
        features=model_params["generator_features"],
        image_size=data_fixed.image_size,
    )
    discriminator = Discriminator(
        channels=data_fixed.channels,
        features=model_params["discriminator_features"],
        image_size=data_fixed.image_size,
        use_spectral_norm=model_fixed.spectral_norm,
    )
    generator.apply(init_dcgan_weights)
    discriminator.apply(init_dcgan_weights)
    return generator, discriminator


# ---------------------------------------------------------------------------
# Diffusion (DDPM with self-contained UNet)
# ---------------------------------------------------------------------------


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = timesteps.device
        exponent = -math.log(10_000.0) * torch.arange(half, device=device, dtype=torch.float32) / max(half - 1, 1)
        frequencies = torch.exp(exponent)
        angles = timesteps.float()[:, None] * frequencies[None, :]
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if self.dim % 2 == 1:
            embedding = nn.functional.pad(embedding, (0, 1))
        return embedding


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=min(8, in_channels), num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, inputs: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(nn.functional.silu(self.norm1(inputs)))
        hidden = hidden + self.time_proj(nn.functional.silu(time_embedding))[:, :, None, None]
        hidden = self.conv2(self.dropout(nn.functional.silu(self.norm2(hidden))))
        return hidden + self.shortcut(inputs)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.op(inputs)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        upsampled = nn.functional.interpolate(inputs, scale_factor=2, mode="nearest")
        return self.op(upsampled)


class UNet(nn.Module):
    def __init__(
        self,
        channels: int,
        base_channels: int,
        channel_mults: tuple[int, ...],
        time_embedding_dim: int,
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )

        self.input_conv = nn.Conv2d(channels, base_channels, kernel_size=3, padding=1)

        encoder_channels = [base_channels]
        current = base_channels
        self.down_blocks = nn.ModuleList()
        for index, mult in enumerate(channel_mults):
            target = base_channels * mult
            self.down_blocks.append(ResidualBlock(current, target, time_embedding_dim))
            self.down_blocks.append(ResidualBlock(target, target, time_embedding_dim))
            encoder_channels.extend([target, target])
            if index < len(channel_mults) - 1:
                self.down_blocks.append(Downsample(target))
                encoder_channels.append(target)
            current = target

        self.middle_blocks = nn.ModuleList(
            [
                ResidualBlock(current, current, time_embedding_dim),
                ResidualBlock(current, current, time_embedding_dim),
            ]
        )

        self.up_blocks = nn.ModuleList()
        for index, mult in enumerate(reversed(channel_mults)):
            target = base_channels * mult
            for block_index in range(3):
                skip = encoder_channels.pop()
                self.up_blocks.append(ResidualBlock(current + skip, target, time_embedding_dim))
                current = target
            if index < len(channel_mults) - 1:
                self.up_blocks.append(Upsample(current))

        self.output_norm = nn.GroupNorm(num_groups=min(8, current), num_channels=current)
        self.output_conv = nn.Conv2d(current, channels, kernel_size=3, padding=1)

    def forward(self, inputs: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_embedding = self.time_mlp(timesteps)
        hidden = self.input_conv(inputs)
        skips = [hidden]

        for module in self.down_blocks:
            if isinstance(module, ResidualBlock):
                hidden = module(hidden, time_embedding)
            else:
                hidden = module(hidden)
            skips.append(hidden)

        for module in self.middle_blocks:
            hidden = module(hidden, time_embedding)

        for module in self.up_blocks:
            if isinstance(module, ResidualBlock):
                hidden = module(torch.cat([hidden, skips.pop()], dim=1), time_embedding)
            else:
                hidden = module(hidden)

        return self.output_conv(nn.functional.silu(self.output_norm(hidden)))


def linear_beta_schedule(timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos(((t + s) / (1 + s)) * math.pi * 0.5).pow(2)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(min=1e-5, max=0.999)


class DiffusionModel(nn.Module):
    def __init__(
        self,
        unet: UNet,
        timesteps: int,
        schedule: NoiseSchedule,
        beta_start: float,
        beta_end: float,
        image_size: int,
        channels: int,
    ):
        super().__init__()
        if schedule == "linear":
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=alphas_cumprod.dtype), alphas_cumprod[:-1]])
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        self.unet = unet
        self.timesteps = timesteps
        self.schedule = schedule
        self.image_size = image_size
        self.channels = channels

        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev.float())
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).float())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod).float())
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas).float())
        self.register_buffer("posterior_variance", posterior_variance.float())

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alpha * x0 + sqrt_one_minus * noise

    def predict_noise(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.unet(x_t, t)

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        noise_prediction = self.predict_noise(x_t, t)
        beta = self.betas[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        sqrt_recip_alpha = self.sqrt_recip_alphas[t][:, None, None, None]

        mean = sqrt_recip_alpha * (x_t - beta / sqrt_one_minus * noise_prediction)
        if int(t.min()) == 0:
            return mean
        variance = self.posterior_variance[t][:, None, None, None]
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(variance) * noise

    @torch.no_grad()
    def sample(self, batch_size: int, device: torch.device, initial_noise: torch.Tensor | None = None) -> torch.Tensor:
        if initial_noise is None:
            shape = (batch_size, self.channels, self.image_size, self.image_size)
            x = torch.randn(shape, device=device)
        else:
            x = initial_noise.to(device)

        for step in reversed(range(self.timesteps)):
            t = torch.full((x.size(0),), step, device=device, dtype=torch.long)
            x = self.p_sample(x, t)
        return x.clamp(-1.0, 1.0)


def build_diffusion(
    model_params: dict[str, Any],
    model_fixed: DiffusionFixedConfig,
    data_fixed: DataFixedConfig,
) -> DiffusionModel:
    channel_mults = tuple(model_params["channel_mults"])
    unet = UNet(
        channels=data_fixed.channels,
        base_channels=model_params["base_channels"],
        channel_mults=channel_mults,
        time_embedding_dim=model_fixed.time_embedding_dim,
    )
    return DiffusionModel(
        unet=unet,
        timesteps=model_params["timesteps"],
        schedule=model_params["schedule"],
        beta_start=model_params["beta_start"],
        beta_end=model_params["beta_end"],
        image_size=data_fixed.image_size,
        channels=data_fixed.channels,
    )


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
