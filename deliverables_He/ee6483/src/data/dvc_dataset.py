"""Dataset / dataloader construction for the Dogs vs Cats task."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder

from .transforms import build_eval_transform, build_train_transform
from ..utils import seed_worker

# ImageFolder sorts class names alphabetically -> cat=0, dog=1, which is exactly
# the label convention required by the submission file (0=cat, 1=dog).
EXPECTED_CLASS_TO_IDX = {"cat": 0, "dog": 1}


class TransformSubset(Dataset):
    """ImageFolder restricted to a list of indices, with its own transform."""

    def __init__(self, base: ImageFolder, indices, transform=None):
        self.base = base
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        path, target = self.base.samples[self.indices[i]]
        img = self.base.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class UnlabelledImageFolder(Dataset):
    """Test folder: flat list of images, id taken from the numeric file name."""

    def __init__(self, root: str | Path, transform=None):
        self.root = Path(root)
        files = [p for p in self.root.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        # ids are the integer stems (1.jpg -> 1). Sorting numerically keeps the
        # order aligned with sampleSubmission.csv even though lexicographic
        # order would give 1, 10, 100, 11 ...
        self.files = sorted(files, key=lambda p: int(p.stem))
        self.ids = [int(p.stem) for p in self.files]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        from PIL import Image

        img = Image.open(self.files[i]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.ids[i]


def check_class_mapping(root: str | Path) -> None:
    probe = ImageFolder(root=root)
    assert probe.class_to_idx == EXPECTED_CLASS_TO_IDX, (
        f"unexpected class mapping {probe.class_to_idx}; submission requires cat=0, dog=1"
    )


def _indices_per_class(targets: np.ndarray, n_per_class: int | None, seed: int) -> list[int]:
    """Nested class-balanced subset: same seed -> smaller set is a subset of bigger."""
    if n_per_class is None:
        return list(range(len(targets)))
    rng = np.random.RandomState(seed)
    picked: list[int] = []
    for c in np.unique(targets):
        idx = np.where(targets == c)[0]
        idx = idx[rng.permutation(len(idx))][:n_per_class]
        picked.extend(idx.tolist())
    return sorted(picked)


def build_train_loader(
    root: str | Path,
    img_size: int = 224,
    aug: str = "full",
    batch_size: int = 64,
    num_workers: int = 4,
    n_per_class: int | None = None,
    seed: int = 42,
    shuffle: bool = True,
    drop_last: bool = True,
    sampler=None,
) -> tuple[DataLoader, ImageFolder]:
    base = ImageFolder(root=str(root))
    assert base.class_to_idx == EXPECTED_CLASS_TO_IDX, base.class_to_idx
    idx = _indices_per_class(np.array(base.targets), n_per_class, seed)
    ds = TransformSubset(base, idx, build_train_transform(img_size, aug))
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=num_workers > 0,
    )
    return loader, base


def build_eval_loader(
    root: str | Path,
    img_size: int = 224,
    batch_size: int = 128,
    num_workers: int = 4,
    n_per_class: int | None = None,
    seed: int = 42,
) -> tuple[DataLoader, ImageFolder]:
    base = ImageFolder(root=str(root))
    assert base.class_to_idx == EXPECTED_CLASS_TO_IDX, base.class_to_idx
    idx = _indices_per_class(np.array(base.targets), n_per_class, seed)
    ds = TransformSubset(base, idx, build_eval_transform(img_size))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return loader, base


def build_test_loader(
    root: str | Path, img_size: int = 224, batch_size: int = 128, num_workers: int = 4
) -> DataLoader:
    ds = UnlabelledImageFolder(root, build_eval_transform(img_size))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
