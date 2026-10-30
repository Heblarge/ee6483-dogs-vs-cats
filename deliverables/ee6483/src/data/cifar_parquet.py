"""CIFAR-10 loader backed by the Hugging Face parquet mirror.

The official cs.toronto.edu mirror runs at ~50 kB/s from this machine (about one
hour for 170 MB), while the HF mirror delivers ~12 MB/s, so we read the two
parquet shards instead of the original tar.gz. The interface mirrors
`torchvision.datasets.CIFAR10` (`.targets`, `.data`-free `__getitem__`) so the
rest of the codebase does not care which source was used.
"""
from __future__ import annotations

import io
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image
from torch.utils.data import Dataset

CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


class CIFAR10Parquet(Dataset):
    def __init__(self, parquet_path: str | Path, transform=None):
        table = pq.read_table(str(parquet_path), columns=["img", "label"]).to_pydict()
        self.images = [rec["bytes"] for rec in table["img"]]
        self.targets = [int(v) for v in table["label"]]
        self.transform = transform
        self.classes = CIFAR_CLASSES

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        img = Image.open(io.BytesIO(self.images[i])).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[i]


def get_cifar(root: str | Path, train: bool, transform=None):
    root = Path(root)
    name = "train.parquet" if train else "test.parquet"
    p = root / name
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Download it with:\n"
            f"  curl -L -o {p} https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{name}"
        )
    return CIFAR10Parquet(p, transform)
