"""猫狗数据读取。目录约定：<root>/{train,val}/{cat,dog}/*.jpg，<root>/test/<id>.jpg。

标签固定为 cat=0、dog=1。可通过 splits/*.csv 使用固定的子集清单（见 scripts/make_split.py）。
"""
import csv
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..utils.seed import seed_worker
from .transforms import build_transforms

CLASSES = ("cat", "dog")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_labeled(root, split):
    """返回 [(path, label), ...]，按文件名排序保证顺序稳定。"""
    samples = []
    for label, name in enumerate(CLASSES):
        folder = Path(root) / split / name
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)
        samples += [(str(p), label) for p in files]
    return samples


def list_test(root):
    """返回 [(path, id), ...]，按数字 id 升序。"""
    files = [p for p in (Path(root) / "test").iterdir() if p.suffix.lower() in IMG_EXTS]
    files.sort(key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    return [(str(p), p.stem) for p in files]


def subsample_per_class(samples, per_class, seed=0):
    """每类随机抽 per_class 张（固定种子），用于小样本调试或算力不足时的正式子集。"""
    rng = random.Random(seed)
    out = []
    for label in range(len(CLASSES)):
        group = [s for s in samples if s[1] == label]
        out += sorted(rng.sample(group, min(per_class, len(group))))
    return out


def save_split(samples, path, root):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["relpath", "label"])
        for p, y in samples:
            w.writerow([Path(p).relative_to(root).as_posix(), y])


def load_split(path, root):
    with open(path, newline="", encoding="utf-8") as f:
        return [(str(Path(root) / r["relpath"]), int(r["label"])) for r in csv.DictReader(f)]


class ImageListDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def _samples_for(data_cfg, split):
    root = data_cfg["root"]
    split_file = data_cfg.get(f"{split}_split")
    if split_file:
        return load_split(split_file, root)
    samples = list_labeled(root, split)
    per_class = data_cfg.get(f"{split}_per_class")
    if per_class:
        samples = subsample_per_class(samples, per_class, data_cfg.get("split_seed", 0))
    return samples


def _loader(dataset, data_cfg, shuffle, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    workers = data_cfg.get("num_workers", 4)
    return DataLoader(
        dataset,
        batch_size=data_cfg.get("batch_size", 64),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=g,
    )


def build_dataloaders(cfg):
    data_cfg = cfg["data"]
    seed = cfg.get("seed", 0)
    train_ds = ImageListDataset(_samples_for(data_cfg, "train"), build_transforms(data_cfg, train=True))
    val_ds = ImageListDataset(_samples_for(data_cfg, "val"), build_transforms(data_cfg, train=False))
    return _loader(train_ds, data_cfg, True, seed), _loader(val_ds, data_cfg, False, seed)


def build_val_loader(cfg):
    data_cfg = cfg["data"]
    ds = ImageListDataset(_samples_for(data_cfg, "val"), build_transforms(data_cfg, train=False))
    return _loader(ds, data_cfg, False)


def build_test_loader(cfg):
    data_cfg = cfg["data"]
    ds = ImageListDataset(list_test(data_cfg["root"]), build_transforms(data_cfg, train=False))
    return _loader(ds, data_cfg, False)
