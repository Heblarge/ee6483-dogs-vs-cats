"""Model zoo: classical baseline, small CNN, and torchvision backbones."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def build_backbone(name: str, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    name = name.lower()
    weights = "DEFAULT" if pretrained else None

    if name == "resnet18":
        m = models.resnet18(weights=weights)
    elif name == "resnet34":
        m = models.resnet34(weights=weights)
    elif name == "resnet50":
        m = models.resnet50(weights=weights)
    elif name == "vgg16":
        m = models.vgg16_bn(weights=weights)
    else:
        raise ValueError(f"unknown backbone {name}")

    if name.startswith("resnet"):
        in_f = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_f, num_classes))
    elif name == "vgg16":
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
    return m


def build_cifar_resnet(name: str = "resnet18", num_classes: int = 10) -> nn.Module:
    """CIFAR variant: 3x3 stride-1 stem, no maxpool.

    The ImageNet stem (7x7/2 + maxpool) reduces a 32x32 input to 8x8 before the
    first residual stage, throwing away most of the signal -- this is the single
    most important architectural change when moving from Dogs vs Cats to CIFAR-10.
    """
    import torchvision.models.resnet as R

    fn = {"resnet18": R.resnet18, "resnet34": R.resnet34}[name]
    m = fn(weights=None, num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


def head_parameters(model: nn.Module):
    """Return the parameters of the final classification layer."""
    if isinstance(model, models.ResNet):
        return list(model.fc.parameters())
    if hasattr(model, "classifier"):
        return list(model.classifier[-1].parameters())
    raise ValueError("cannot locate classification head")


def split_backbone_head(model: nn.Module):
    head_ids = {id(p) for p in head_parameters(model)}
    head = [p for p in model.parameters() if id(p) in head_ids]
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    return backbone, head


def freeze_backbone(model: nn.Module) -> None:
    for p in split_backbone_head(model)[0]:
        p.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


def set_stage_trainable(model: nn.Module, trainable_prefixes: tuple[str, ...]) -> None:
    """Unfreeze only modules whose name starts with one of the prefixes."""
    for name, p in model.named_parameters():
        p.requires_grad = any(name.startswith(pre) for pre in trainable_prefixes)
    for p in head_parameters(model):
        p.requires_grad = True


def get_layer_groups(model: nn.Module):
    """Split params into (backbone, head) for differential learning rates."""
    return split_backbone_head(model)
