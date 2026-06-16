import math
import statistics
import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .config import Device, GANFixedConfig, TrainFixedConfig, VAEFixedConfig
from .models import DiffusionModel, Discriminator, Generator, VAE, count_parameters
from .outputs import save_checkpoint, save_sample_grid
from .progress import progress_bar, stage


def cuda_supports_current_device() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        x = torch.ones(1, device="cuda")
        y = x + 1
        torch.cuda.synchronize()
    except RuntimeError:
        return False
    return y.is_cuda


def resolve_device(device: Device) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if cuda_supports_current_device() else "cpu")
    if device == "cuda" and not cuda_supports_current_device():
        raise RuntimeError(
            "CUDA was requested, but the installed PyTorch build does not support this GPU."
        )
    return torch.device(device)


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float, restore_best: bool):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best: float | None = None
        self.best_epoch: int | None = None
        self.counter = 0
        self.best_state: dict[str, torch.Tensor] | None = None

    def step(self, metric: float, model: nn.Module, epoch: int) -> bool:
        # Improvement must beat the previous best by at least min_delta — small
        # noise-level changes do not reset the patience counter.
        improved = self.best is None or metric < self.best - self.min_delta
        if improved:
            self.best = metric
            self.best_epoch = epoch
            self.counter = 0
            if self.restore_best:
                self.best_state = {
                    name: tensor.detach().clone().cpu()
                    for name, tensor in model.state_dict().items()
                }
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module) -> None:
        if self.best_state is None:
            return
        device = next(model.parameters()).device
        model.load_state_dict(
            {name: tensor.to(device) for name, tensor in self.best_state.items()}
        )


def _build_early_stopper(train_fixed: TrainFixedConfig) -> EarlyStopping | None:
    if not train_fixed.early_stopping:
        return None
    return EarlyStopping(
        patience=train_fixed.early_stopping_patience,
        min_delta=train_fixed.early_stopping_min_delta,
        restore_best=train_fixed.early_stopping_restore_best,
    )


# ---------------------------------------------------------------------------
# VAE training
# ---------------------------------------------------------------------------


def effective_beta(target_beta: float, epoch: int, anneal_epochs: int) -> float:
    # Linear warmup from 0 to target_beta over `anneal_epochs` epochs.
    # Pure function of the absolute epoch number — when resuming from a
    # checkpoint at epoch N, calling with epoch=N+1 picks up at the right
    # annealed value with no extra state.
    if anneal_epochs <= 0:
        return target_beta
    return target_beta * min(1.0, epoch / anneal_epochs)


def vae_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> dict[str, torch.Tensor]:
    batch_size = target.size(0)
    reconstruction_loss = F.mse_loss(reconstruction, target, reduction="sum") / batch_size
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    return {
        "loss": reconstruction_loss + beta * kl_loss,
        "reconstruction": reconstruction_loss,
        "kl": kl_loss,
    }


def _amp_autocast(device: torch.device, enabled: bool) -> Callable[[], Any]:
    if enabled and device.type == "cuda":
        return lambda: torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext


def train_vae_one_epoch(
    model: VAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    beta: float,
    scaler: "torch.amp.GradScaler | None" = None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_examples = 0
    amp_enabled = scaler is not None
    autocast_ctx = _amp_autocast(device, amp_enabled)

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx():
            reconstruction, mu, logvar = model(batch)
            losses = vae_loss(reconstruction, batch, mu, logvar, beta)

        if amp_enabled:
            scaler.scale(losses["loss"]).backward()
            # Unscale before clipping so max_norm applies to the real gradient
            # magnitudes, not the loss-scaler-inflated ones.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        examples = batch.size(0)
        total_loss += losses["loss"].item() * examples
        total_recon += losses["reconstruction"].item() * examples
        total_kl += losses["kl"].item() * examples
        total_examples += examples

    return {
        "loss": total_loss / total_examples,
        "reconstruction": total_recon / total_examples,
        "kl": total_kl / total_examples,
    }


@torch.no_grad()
def evaluate_vae(
    model: VAE,
    loader: DataLoader,
    device: torch.device,
    beta: float,
    amp_enabled: bool = False,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_examples = 0
    autocast_ctx = _amp_autocast(device, amp_enabled)

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with autocast_ctx():
            reconstruction, mu, logvar = model(batch)
            losses = vae_loss(reconstruction, batch, mu, logvar, beta)
        examples = batch.size(0)
        total_loss += losses["loss"].item() * examples
        total_recon += losses["reconstruction"].item() * examples
        total_kl += losses["kl"].item() * examples
        total_examples += examples

    return {
        "validation_loss": total_loss / total_examples,
        "validation_reconstruction": total_recon / total_examples,
        "validation_kl": total_kl / total_examples,
    }


def train_vae(
    model: VAE,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    model_params: dict[str, Any],
    train_params: dict[str, Any],
    train_fixed: TrainFixedConfig,
    fixed_params: VAEFixedConfig | None = None,
    sample_dir: Path | None = None,
    checkpoint_path: Path | None = None,
    resume_from: Path | str | None = None,
) -> dict[str, Any]:
    device = resolve_device(train_fixed.device)
    stage(f"Using device: {device}", enabled=train_fixed.verbose)
    stage(
        f"VAE parameters: {count_parameters(model):,}",
        enabled=train_fixed.verbose,
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=model_params["learning_rate"],
        weight_decay=model_params["weight_decay"],
    )

    amp_enabled = (
        fixed_params is not None
        and fixed_params.amp
        and device.type == "cuda"
    )
    scaler = torch.amp.GradScaler("cuda") if amp_enabled else None
    if amp_enabled:
        stage("AMP enabled (fp16 autocast + GradScaler).", enabled=train_fixed.verbose)

    history: list[dict[str, float]] = []
    start_epoch = 0
    if resume_from is not None:
        ckpt = torch.load(Path(resume_from), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt["epoch"])
        history = list(ckpt.get("history", []))
        stage(
            f"Resumed VAE from {resume_from} at epoch {start_epoch}.",
            enabled=train_fixed.verbose,
        )

    sample_every = int(train_params["sample_every"]) if train_params["sample_every"] else 0
    early_stopper = _build_early_stopper(train_fixed)
    stopped_early = False
    last_epoch = start_epoch
    start_time = time.perf_counter()
    epochs = progress_bar(
        range(start_epoch + 1, train_params["epochs"] + 1),
        enabled=train_fixed.use_tqdm,
        backend=train_fixed.progress_backend,
        description="VAE Training",
        leave=True,
    )

    target_beta = float(model_params["beta"])
    anneal_epochs = int(model_params.get("kl_anneal_epochs", 0) or 0)
    for epoch in epochs:
        last_epoch = epoch
        beta_now = effective_beta(target_beta, epoch, anneal_epochs)
        train_metrics = train_vae_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            beta=beta_now,
            scaler=scaler,
        )
        validation_metrics = evaluate_vae(
            model,
            validation_loader,
            device,
            beta=beta_now,
            amp_enabled=amp_enabled,
        )
        row = {"epoch": epoch, "beta": beta_now, **train_metrics, **validation_metrics}
        history.append(row)
        if train_fixed.use_tqdm:
            epochs.set_postfix(
                loss=f"{row['loss']:.3f}",
                val=f"{row['validation_loss']:.3f}",
                kl=f"{row['kl']:.3f}",
                beta=f"{beta_now:.3f}",
                refresh=False,
            )

        if sample_dir is not None and sample_every and epoch % sample_every == 0:
            save_vae_samples(model, device, sample_dir / f"epoch_{epoch:03d}.png")

        if early_stopper is not None and early_stopper.step(
            validation_metrics["validation_loss"], model, epoch
        ):
            stopped_early = True
            stage(
                f"Early stopping at epoch {epoch}; best validation_loss="
                f"{early_stopper.best:.4f} at epoch {early_stopper.best_epoch}.",
                enabled=train_fixed.verbose,
            )
            break

    if early_stopper is not None and early_stopper.restore_best:
        early_stopper.restore(model)
        if early_stopper.best_epoch is not None:
            stage(
                f"Restored best VAE weights from epoch {early_stopper.best_epoch}.",
                enabled=train_fixed.verbose,
            )

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            {
                "epoch": last_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "model_params": model_params,
                "train_params": train_params,
            },
            checkpoint_path,
        )
        stage(f"Saved VAE checkpoint to {checkpoint_path}.", enabled=train_fixed.verbose)

    elapsed = time.perf_counter() - start_time
    stage(f"VAE training finished in {int(elapsed // 3600):02d}:{int(elapsed % 3600 // 60):02d}:{int(elapsed % 60):02d}", enabled=train_fixed.verbose)
    return {
        "history": history,
        "elapsed_seconds": elapsed,
        "device": str(device),
        "early_stopped": stopped_early,
        "best_epoch": early_stopper.best_epoch if early_stopper is not None else None,
    }


@torch.no_grad()
def save_vae_samples(model: VAE, device: torch.device, path: Path, num_samples: int = 64) -> None:
    model.eval()
    z = torch.randn(num_samples, model.latent_dim, device=device)
    samples = model.decode(z)
    save_sample_grid(samples, path, nrow=8)


# ---------------------------------------------------------------------------
# GAN training (DCGAN with mode-collapse detection)
# ---------------------------------------------------------------------------


def detect_mode_collapse(
    discriminator_losses: list[float],
    window: int,
    min_std: float,
) -> bool:
    if len(discriminator_losses) < window:
        return False
    recent = discriminator_losses[-window:]
    if len(recent) < 2:
        return False
    return statistics.pstdev(recent) < min_std


def _lr_factor(step: int, warmup_steps: int, cosine_total_steps: int | None) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if cosine_total_steps is None or cosine_total_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / max(cosine_total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _apply_lr(optimizer: torch.optim.Optimizer, base_lr: float, factor: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = base_lr * factor


def train_gan_one_epoch(
    generator: Generator,
    discriminator: Discriminator,
    loader: DataLoader,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    device: torch.device,
    label_smoothing: float,
    n_disc_steps: int,
    fixed_params: GANFixedConfig,
    discriminator_loss_buffer: list[float],
    collapse_alert: Callable[[int], None] | None = None,
    *,
    ema: "EMA | None" = None,
    scaler_g: "torch.amp.GradScaler | None" = None,
    scaler_d: "torch.amp.GradScaler | None" = None,
    base_lr_g: float = 0.0,
    base_lr_d: float = 0.0,
    warmup_steps: int = 0,
    cosine_total_steps: int | None = None,
    step_offset: int = 0,
) -> tuple[dict[str, float], int]:
    generator.train()
    discriminator.train()
    bce = nn.BCEWithLogitsLoss()
    amp_enabled = scaler_g is not None and scaler_d is not None
    autocast_ctx: Callable[[], Any]
    if amp_enabled and device.type == "cuda":
        autocast_ctx = lambda: torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    else:
        autocast_ctx = nullcontext

    total_d = 0.0
    total_g = 0.0
    total_real_logit = 0.0
    total_fake_logit = 0.0
    examples = 0
    collapse_signaled = False
    step = step_offset

    for batch_index, real in enumerate(loader):
        real = real.to(device, non_blocking=True)
        batch_size = real.size(0)

        if base_lr_g > 0.0 or base_lr_d > 0.0:
            factor = _lr_factor(step, warmup_steps, cosine_total_steps)
            if base_lr_g > 0.0:
                _apply_lr(optimizer_g, base_lr_g, factor)
            if base_lr_d > 0.0:
                _apply_lr(optimizer_d, base_lr_d, factor)

        # Discriminator updates.
        generator.eval()
        for _ in range(n_disc_steps):
            optimizer_d.zero_grad(set_to_none=True)
            noise = torch.randn(batch_size, generator.latent_dim, 1, 1, device=device)
            with autocast_ctx():
                fake = generator(noise).detach()
                real_labels = torch.full((batch_size,), 1.0 - label_smoothing, device=device)
                fake_labels = torch.zeros(batch_size, device=device)
                real_logits = discriminator(real)
                fake_logits = discriminator(fake)
                d_loss = bce(real_logits, real_labels) + bce(fake_logits, fake_labels)
            if amp_enabled:
                scaler_d.scale(d_loss).backward()
                scaler_d.step(optimizer_d)
                scaler_d.update()
            else:
                d_loss.backward()
                optimizer_d.step()

        discriminator_loss_buffer.append(d_loss.item())

        # Generator update.
        generator.train()
        optimizer_g.zero_grad(set_to_none=True)
        noise = torch.randn(batch_size, generator.latent_dim, 1, 1, device=device)
        with autocast_ctx():
            fake = generator(noise)
            fake_logits_for_g = discriminator(fake)
            target = torch.ones(batch_size, device=device)
            g_loss = bce(fake_logits_for_g, target)
        if amp_enabled:
            scaler_g.scale(g_loss).backward()
            scaler_g.step(optimizer_g)
            scaler_g.update()
        else:
            g_loss.backward()
            optimizer_g.step()

        if ema is not None:
            ema.update(generator)

        total_d += d_loss.item() * batch_size
        total_g += g_loss.item() * batch_size
        total_real_logit += real_logits.detach().float().mean().item() * batch_size
        total_fake_logit += fake_logits.detach().float().mean().item() * batch_size
        examples += batch_size
        step += 1

        if not collapse_signaled and detect_mode_collapse(
            discriminator_loss_buffer,
            window=fixed_params.mode_collapse_window,
            min_std=fixed_params.mode_collapse_min_std,
        ):
            collapse_signaled = True
            if collapse_alert is not None:
                collapse_alert(batch_index)

    metrics = {
        "d_loss": total_d / examples,
        "g_loss": total_g / examples,
        "real_logit": total_real_logit / examples,
        "fake_logit": total_fake_logit / examples,
        "mode_collapse": float(collapse_signaled),
    }
    return metrics, step


def train_gan(
    generator: Generator,
    discriminator: Discriminator,
    train_loader: DataLoader,
    model_params: dict[str, Any],
    train_params: dict[str, Any],
    train_fixed: TrainFixedConfig,
    fixed_params: GANFixedConfig,
    sample_dir: Path | None = None,
    checkpoint_path: Path | None = None,
    resume_from: Path | str | None = None,
) -> dict[str, Any]:
    device = resolve_device(train_fixed.device)
    stage(f"Using device: {device}", enabled=train_fixed.verbose)
    stage(
        f"Generator parameters: {count_parameters(generator):,}; "
        f"Discriminator parameters: {count_parameters(discriminator):,}",
        enabled=train_fixed.verbose,
    )
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=model_params["learning_rate_g"],
        betas=(model_params["beta1"], model_params["beta2"]),
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=model_params["learning_rate_d"],
        betas=(model_params["beta1"], model_params["beta2"]),
    )

    amp_enabled = fixed_params.amp and device.type == "cuda"
    scaler_g = torch.amp.GradScaler("cuda") if amp_enabled else None
    scaler_d = torch.amp.GradScaler("cuda") if amp_enabled else None
    ema = EMA(generator, decay=fixed_params.ema_decay) if fixed_params.ema_decay > 0 else None

    history: list[dict[str, float]] = []
    discriminator_loss_buffer: list[float] = []
    collapse_alerts: list[dict[str, int]] = []
    start_epoch = 0
    if resume_from is not None:
        ckpt = torch.load(Path(resume_from), map_location=device)
        generator.load_state_dict(ckpt["generator_state_dict"])
        discriminator.load_state_dict(ckpt["discriminator_state_dict"])
        optimizer_g.load_state_dict(ckpt["optimizer_g_state_dict"])
        optimizer_d.load_state_dict(ckpt["optimizer_d_state_dict"])
        start_epoch = int(ckpt["epoch"])
        history = list(ckpt.get("history", []))
        discriminator_loss_buffer = list(ckpt.get("discriminator_loss_buffer", []))
        collapse_alerts = list(ckpt.get("collapse_alerts", []))
        if ema is not None and ckpt.get("ema_shadow") is not None:
            ema.shadow = {
                name: tensor.to(device)
                for name, tensor in ckpt["ema_shadow"].items()
            }
        stage(
            f"Resumed GAN from {resume_from} at epoch {start_epoch}.",
            enabled=train_fixed.verbose,
        )

    def on_collapse(batch_index: int) -> None:
        collapse_alerts.append({"epoch": int(epoch), "batch": int(batch_index)})
        stage(
            f"\n[mode collapse] discriminator loss variance below "
            f"{fixed_params.mode_collapse_min_std} at epoch {epoch}, batch {batch_index}.",
            enabled=train_fixed.verbose,
        )

    sample_every = int(train_params["sample_every"]) if train_params["sample_every"] else 0
    last_epoch = start_epoch
    start_time = time.perf_counter()
    steps_per_epoch = len(train_loader)
    total_steps = train_params["epochs"] * steps_per_epoch
    cosine_total = total_steps if fixed_params.lr_cosine_decay else None
    step_counter = start_epoch * steps_per_epoch
    epochs = progress_bar(
        range(start_epoch + 1, train_params["epochs"] + 1),
        enabled=train_fixed.use_tqdm,
        backend=train_fixed.progress_backend,
        description="GAN Training",
        leave=True,
    )

    for epoch in epochs:
        last_epoch = epoch
        metrics, step_counter = train_gan_one_epoch(
            generator,
            discriminator,
            train_loader,
            optimizer_g,
            optimizer_d,
            device,
            label_smoothing=model_params["label_smoothing"],
            n_disc_steps=model_params["n_disc_steps"],
            fixed_params=fixed_params,
            discriminator_loss_buffer=discriminator_loss_buffer,
            collapse_alert=on_collapse,
            ema=ema,
            scaler_g=scaler_g,
            scaler_d=scaler_d,
            base_lr_g=model_params["learning_rate_g"],
            base_lr_d=model_params["learning_rate_d"],
            warmup_steps=fixed_params.lr_warmup_steps,
            cosine_total_steps=cosine_total,
            step_offset=step_counter,
        )
        row = {"epoch": epoch, **metrics}
        history.append(row)
        if train_fixed.use_tqdm:
            epochs.set_postfix(
                d=f"{row['d_loss']:.3f}",
                g=f"{row['g_loss']:.3f}",
                rl=f"{row['real_logit']:.2f}",
                fl=f"{row['fake_logit']:.2f}",
                refresh=False,
            )

        if sample_dir is not None and sample_every and epoch % sample_every == 0:
            sampling_model = generator
            if ema is not None:
                # snapshot live params, swap in EMA weights for sampling, then restore.
                live_state = {name: p.detach().clone() for name, p in generator.named_parameters()}
                ema.copy_to(generator)
                save_gan_samples(generator, device, sample_dir / f"epoch_{epoch:03d}.png")
                with torch.no_grad():
                    for name, p in generator.named_parameters():
                        if name in live_state:
                            p.data.copy_(live_state[name])
            else:
                save_gan_samples(sampling_model, device, sample_dir / f"epoch_{epoch:03d}.png")

    if ema is not None:
        ema.copy_to(generator)
        stage("Copied EMA weights into generator for evaluation.", enabled=train_fixed.verbose)

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            {
                "epoch": last_epoch,
                "generator_state_dict": generator.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "optimizer_g_state_dict": optimizer_g.state_dict(),
                "optimizer_d_state_dict": optimizer_d.state_dict(),
                "ema_shadow": (
                    {name: tensor.detach().cpu().clone() for name, tensor in ema.shadow.items()}
                    if ema is not None
                    else None
                ),
                "history": history,
                "collapse_alerts": collapse_alerts,
                "discriminator_loss_buffer": discriminator_loss_buffer,
                "model_params": model_params,
                "train_params": train_params,
            },
            checkpoint_path,
        )
        stage(f"Saved GAN checkpoint to {checkpoint_path}.", enabled=train_fixed.verbose)

    elapsed = time.perf_counter() - start_time
    stage(f"GAN training finished in {int(elapsed // 3600):02d}:{int(elapsed % 3600 // 60):02d}:{int(elapsed % 60):02d}", enabled=train_fixed.verbose)
    return {
        "history": history,
        "elapsed_seconds": elapsed,
        "device": str(device),
        "mode_collapse_events": collapse_alerts,
    }


@torch.no_grad()
def save_gan_samples(generator: Generator, device: torch.device, path: Path, num_samples: int = 64) -> None:
    generator.eval()
    noise = torch.randn(num_samples, generator.latent_dim, 1, 1, device=device)
    samples = generator(noise)
    save_sample_grid(samples, path, nrow=8)


# ---------------------------------------------------------------------------
# Diffusion training (DDPM)
# ---------------------------------------------------------------------------


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            self.shadow[name].mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in self.shadow:
                    parameter.data.copy_(self.shadow[name])


def train_diffusion_one_epoch(
    model: DiffusionModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    ema: EMA | None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_examples = 0

    for batch in loader:
        batch = batch.to(device)
        batch_size = batch.size(0)
        timesteps = torch.randint(0, model.timesteps, (batch_size,), device=device, dtype=torch.long)
        noise = torch.randn_like(batch)
        noisy = model.q_sample(batch, timesteps, noise)
        predicted = model.predict_noise(noisy, timesteps)
        loss = F.mse_loss(predicted, noise)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return {"loss": total_loss / total_examples}


@torch.no_grad()
def evaluate_diffusion(
    model: DiffusionModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_examples = 0

    for batch in loader:
        batch = batch.to(device)
        batch_size = batch.size(0)
        timesteps = torch.randint(0, model.timesteps, (batch_size,), device=device, dtype=torch.long)
        noise = torch.randn_like(batch)
        noisy = model.q_sample(batch, timesteps, noise)
        predicted = model.predict_noise(noisy, timesteps)
        loss = F.mse_loss(predicted, noise)

        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return {"validation_loss": total_loss / total_examples}


def train_diffusion(
    model: DiffusionModel,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    model_params: dict[str, Any],
    train_params: dict[str, Any],
    train_fixed: TrainFixedConfig,
    sample_dir: Path | None = None,
    checkpoint_path: Path | None = None,
    resume_from: Path | str | None = None,
) -> dict[str, Any]:
    device = resolve_device(train_fixed.device)
    stage(f"Using device: {device}", enabled=train_fixed.verbose)
    stage(
        f"Diffusion parameters: {count_parameters(model):,}",
        enabled=train_fixed.verbose,
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_params["learning_rate"])
    ema_decay = float(model_params.get("ema_decay", 0.0))
    ema = EMA(model, decay=ema_decay) if ema_decay > 0 else None

    history: list[dict[str, float]] = []
    start_epoch = 0
    if resume_from is not None:
        ckpt = torch.load(Path(resume_from), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt["epoch"])
        history = list(ckpt.get("history", []))
        if ema is not None and ckpt.get("ema_shadow") is not None:
            ema.shadow = {
                name: tensor.to(device)
                for name, tensor in ckpt["ema_shadow"].items()
            }
        stage(
            f"Resumed diffusion from {resume_from} at epoch {start_epoch}.",
            enabled=train_fixed.verbose,
        )

    sample_every = int(train_params["sample_every"]) if train_params["sample_every"] else 0
    early_stopper = _build_early_stopper(train_fixed)
    stopped_early = False
    last_epoch = start_epoch
    start_time = time.perf_counter()
    epochs = progress_bar(
        range(start_epoch + 1, train_params["epochs"] + 1),
        enabled=train_fixed.use_tqdm,
        backend=train_fixed.progress_backend,
        description="Diffusion Training",
        leave=True,
    )

    for epoch in epochs:
        last_epoch = epoch
        train_metrics = train_diffusion_one_epoch(
            model, train_loader, optimizer, device, ema,
        )
        validation_metrics = evaluate_diffusion(model, validation_loader, device)
        row = {"epoch": epoch, **train_metrics, **validation_metrics}
        history.append(row)
        if train_fixed.use_tqdm:
            epochs.set_postfix(
                loss=f"{row['loss']:.4f}",
                val=f"{row['validation_loss']:.4f}",
                refresh=False,
            )

        if sample_dir is not None and sample_every and epoch % sample_every == 0:
            save_diffusion_samples(model, device, sample_dir / f"epoch_{epoch:03d}.png")

        if early_stopper is not None and early_stopper.step(
            validation_metrics["validation_loss"], model, epoch
        ):
            stopped_early = True
            stage(
                f"Early stopping at epoch {epoch}; best validation_loss="
                f"{early_stopper.best:.4f} at epoch {early_stopper.best_epoch}.",
                enabled=train_fixed.verbose,
            )
            break

    # When restore_best is active, restoring validation-best model weights takes
    # precedence over copying in the EMA shadow (which was never validated).
    if early_stopper is not None and early_stopper.restore_best:
        early_stopper.restore(model)
        if early_stopper.best_epoch is not None:
            stage(
                f"Restored best diffusion weights from epoch {early_stopper.best_epoch}.",
                enabled=train_fixed.verbose,
            )
    elif ema is not None:
        ema.copy_to(model)

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            {
                "epoch": last_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "ema_shadow": (
                    {name: tensor.detach().cpu().clone() for name, tensor in ema.shadow.items()}
                    if ema is not None
                    else None
                ),
                "history": history,
                "model_params": model_params,
                "train_params": train_params,
            },
            checkpoint_path,
        )
        stage(f"Saved diffusion checkpoint to {checkpoint_path}.", enabled=train_fixed.verbose)

    elapsed = time.perf_counter() - start_time
    stage(f"Diffusion training finished in {int(elapsed // 3600):02d}:{int(elapsed % 3600 // 60):02d}:{int(elapsed % 60):02d}", enabled=train_fixed.verbose)
    return {
        "history": history,
        "elapsed_seconds": elapsed,
        "device": str(device),
        "early_stopped": stopped_early,
        "best_epoch": early_stopper.best_epoch if early_stopper is not None else None,
    }


@torch.no_grad()
def save_diffusion_samples(
    model: DiffusionModel,
    device: torch.device,
    path: Path,
    num_samples: int = 16,
) -> None:
    model.eval()
    samples = model.sample(num_samples, device)
    save_sample_grid(samples, path, nrow=4)
