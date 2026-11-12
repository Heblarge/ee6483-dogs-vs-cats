"""Data augmentation / preprocessing pipelines.

Two ideas drive the design:
  1. train pipeline is stochastic, eval pipeline is fully deterministic
     (Resize -> CenterCrop), so val/test numbers are comparable across runs;
  2. augmentation strength is a *named level* so the ablation in Part F can
     turn it up one step at a time instead of inventing new code each time.
"""
from __future__ import annotations

from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# aug levels, from weakest to strongest. Each level is a superset of the previous.
AUG_LEVELS = ["none", "flip", "geo", "color", "full", "mix"]


def build_train_transform(
    size: int = 224,
    aug: str = "full",
    crop_scale: tuple[float, float] = (0.5, 1.0),
    rotation: int = 15,
) -> transforms.Compose:
    if aug not in AUG_LEVELS:
        raise ValueError(f"aug must be one of {AUG_LEVELS}, got {aug}")

    ops: list = []
    if aug == "none":
        # still need a deterministic geometry op, otherwise batches of
        # differently-sized images cannot be stacked
        ops += [transforms.Resize(int(size * 1.14)), transforms.CenterCrop(size)]
    else:
        ops += [transforms.RandomResizedCrop(size, scale=crop_scale)]

    if aug in ("flip", "geo", "color", "full", "mix"):
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
    if aug in ("geo", "color", "full", "mix"):
        ops.append(transforms.RandomRotation(rotation))
    if aug in ("color", "full", "mix"):
        ops.append(transforms.ColorJitter(0.2, 0.2, 0.2, 0.05))
    if aug in ("full", "mix"):
        ops.append(transforms.RandAugment(num_ops=2, magnitude=7))

    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]

    if aug in ("full", "mix"):
        # random erasing fights the background shortcut (grass / sofa)
        ops.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)))
    return transforms.Compose(ops)


def build_eval_transform(size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(int(round(size * 1.14))),  # 224 -> 256, keeps ratio
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_cifar_train_transform(aug: str = "full") -> transforms.Compose:
    ops: list = []
    if aug != "none":
        ops += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if aug in ("full", "mix"):
        ops.append(transforms.RandAugment(num_ops=2, magnitude=9))
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if aug in ("full", "mix"):
        ops.append(transforms.RandomErasing(p=0.25))
    return transforms.Compose(ops)


def build_cifar_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )


def build_upsample_transform(size: int, train: bool = False, aug: str = "full"):
    """CIFAR 32x32 -> `size` upsampling, used by the pretrained CIFAR model."""
    ops: list = [transforms.Resize(size, antialias=True)]
    if train:
        ops += [transforms.RandomCrop(size, padding=size // 8), transforms.RandomHorizontalFlip()]
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if train and aug in ("full", "mix"):
        ops.append(transforms.RandomErasing(p=0.25))
    return transforms.Compose(ops)
