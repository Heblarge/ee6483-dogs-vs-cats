"""Data pipeline for Dogs vs Cats (binary) and CIFAR-10 (10-way).

Notes
-----
* ``ImageFolder`` sorts class names alphabetically -> cat=0, dog=1, which matches
  the submission convention required by the brief (1=dog, 0=cat).
* Test images are flat files (``1.jpg`` ... ``500.jpg``); ``TestDataset`` sorts
  them numerically so that row order is predictable.
* ``seed_worker`` + a dedicated generator make DataLoader augmentation
  reproducible across runs (the brief asks for reproducibility).
"""
import os
import random
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


def build_train_transform(img_size=160, strong=False):
    ops = [
        transforms.RandomResizedCrop(img_size, scale=(0.65, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]
    if strong:
        ops += [
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
        ]
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)


def build_eval_transform(img_size=160):
    return transforms.Compose([
        transforms.Resize(int(round(img_size * 1.14))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class TestDataset(Dataset):
    """Flat image directory; returns ``(tensor, id_str)``."""

    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.files = sorted((f for f in os.listdir(root)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))),
                            key=self._key)

    @staticmethod
    def _key(name):
        stem = os.path.splitext(name)[0]
        return (0, int(stem)) if stem.isdigit() else (1, stem)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fn = self.files[idx]
        img = Image.open(os.path.join(self.root, fn)).convert('RGB')
        return self.transform(img), os.path.splitext(fn)[0]


def _subsample(dataset, per_class):
    """Keep the first ``per_class`` samples of every class (deterministic)."""
    if per_class is None:
        return dataset
    idx_by_class = defaultdict(list)
    for i, t in enumerate(dataset.targets):
        idx_by_class[t].append(i)
    keep = []
    for _, idxs in sorted(idx_by_class.items()):
        keep.extend(idxs[:per_class])
    return Subset(dataset, keep)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def make_loader(ds, batch_size, shuffle, num_workers=4, seed=42, sampler=None):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle if sampler is None else False,
        sampler=sampler, num_workers=num_workers, pin_memory=True,
        drop_last=False, worker_init_fn=seed_worker, generator=g,
        persistent_workers=num_workers > 0,
    )


def get_dataloaders(data_root='dataset', img_size=160, batch_size=64, num_workers=4,
                    strong=False, max_train_per_class=None, max_val_per_class=None,
                    seed=42):
    train_ds = datasets.ImageFolder(os.path.join(data_root, 'train'),
                                    transform=build_train_transform(img_size, strong))
    val_ds = datasets.ImageFolder(os.path.join(data_root, 'val'),
                                  transform=build_eval_transform(img_size))
    class_names = train_ds.classes  # ['cat', 'dog']

    train_ds = _subsample(train_ds, max_train_per_class)
    val_ds = _subsample(val_ds, max_val_per_class)

    train_ld = make_loader(train_ds, batch_size, True, num_workers, seed)
    val_ld = make_loader(val_ds, batch_size, False, num_workers, seed)

    test_ld = None
    test_dir = os.path.join(data_root, 'test')
    if os.path.isdir(test_dir):
        test_ds = TestDataset(test_dir, build_eval_transform(img_size))
        test_ld = make_loader(test_ds, batch_size, False, num_workers, seed)

    return train_ld, val_ld, test_ld, train_ds, val_ds, class_names
