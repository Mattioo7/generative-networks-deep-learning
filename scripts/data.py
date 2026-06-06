from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import DataFixedConfig

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def list_image_files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    files = [path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS]
    if not files:
        raise FileNotFoundError(
            f"No images with extensions {IMAGE_EXTENSIONS} found under {root}"
        )
    files.sort()
    return files


def build_train_transform(
    image_size: int,
    augment_flip: bool,
    grayscale: bool = False,
    posterize_bits: int | None = None,
) -> transforms.Compose:
    steps: list[object] = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
    ]
    if grayscale:
        steps.append(transforms.Grayscale(num_output_channels=1))
    if posterize_bits is not None:
        steps.append(transforms.RandomPosterize(bits=posterize_bits, p=1.0))
    if augment_flip:
        steps.append(transforms.RandomHorizontalFlip(p=0.5))
    channels = 1 if grayscale else 3
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * channels, (0.5,) * channels),
        ]
    )
    return transforms.Compose(steps)


def build_eval_transform(
    image_size: int,
    grayscale: bool = False,
    posterize_bits: int | None = None,
) -> transforms.Compose:
    steps: list[object] = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
    ]
    if grayscale:
        steps.append(transforms.Grayscale(num_output_channels=1))
    if posterize_bits is not None:
        steps.append(transforms.RandomPosterize(bits=posterize_bits, p=1.0))
    channels = 1 if grayscale else 3
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * channels, (0.5,) * channels),
        ]
    )
    return transforms.Compose(steps)


class CatDataset(Dataset):
    def __init__(
        self,
        files: list[Path],
        transform: transforms.Compose,
    ):
        self.files = files
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.files[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            return self.transform(image)


@dataclass(frozen=True)
class PreparedData:
    train_dataset: Dataset
    validation_dataset: Dataset
    train_indices: list[int]
    validation_indices: list[int]
    files: list[Path]


def split_dataset(
    files: list[Path],
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if train_fraction + validation_fraction > 1.0 + 1e-6:
        raise ValueError(
            f"train_fraction ({train_fraction}) + validation_fraction ({validation_fraction}) > 1"
        )

    total = len(files)
    train_size = int(round(total * train_fraction))
    validation_size = int(round(total * validation_fraction))
    if train_size == 0:
        raise ValueError("Empty training split. Increase train_fraction or dataset size.")

    train_size = min(train_size, total - max(validation_size, 0))
    validation_size = min(validation_size, total - train_size)

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(total, generator=generator).tolist()
    train_indices = permutation[:train_size]
    validation_indices = permutation[train_size : train_size + validation_size]
    return train_indices, validation_indices


def prepare_data(
    data_fixed: DataFixedConfig,
    train_fraction: float,
    validation_fraction: float,
    augment_flip: bool,
    seed: int,
) -> PreparedData:
    files = list_image_files(Path(data_fixed.data_dir))
    train_indices, validation_indices = split_dataset(
        files,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        seed=seed,
    )

    train_files = [files[i] for i in train_indices]
    validation_files = [files[i] for i in validation_indices]

    train_dataset = CatDataset(
        train_files,
        build_train_transform(
            data_fixed.image_size,
            augment_flip,
            grayscale=data_fixed.grayscale,
            posterize_bits=data_fixed.posterize_bits,
        ),
    )
    validation_dataset = CatDataset(
        validation_files,
        build_eval_transform(
            data_fixed.image_size,
            grayscale=data_fixed.grayscale,
            posterize_bits=data_fixed.posterize_bits,
        ),
    )

    return PreparedData(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        train_indices=train_indices,
        validation_indices=validation_indices,
        files=files,
    )


def build_dataloaders(
    prepared: PreparedData,
    data_fixed: DataFixedConfig,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        prepared.train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_fixed.num_workers,
        pin_memory=data_fixed.pin_memory,
        drop_last=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        prepared.validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_fixed.num_workers,
        pin_memory=data_fixed.pin_memory,
        drop_last=False,
    )
    return train_loader, validation_loader


def real_image_loader(
    data_fixed: DataFixedConfig,
    num_samples: int,
    batch_size: int,
    seed: int,
) -> DataLoader:
    files = list_image_files(Path(data_fixed.data_dir))
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(files), generator=generator).tolist()
    selected = [files[i] for i in permutation[:num_samples]]
    dataset = CatDataset(
        selected,
        build_eval_transform(
            data_fixed.image_size,
            grayscale=data_fixed.grayscale,
            posterize_bits=data_fixed.posterize_bits,
        ),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_fixed.num_workers,
        pin_memory=data_fixed.pin_memory,
        drop_last=False,
    )
