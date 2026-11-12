"""ConvNeXt：借鉴 Transformer 设计的现代纯卷积网络（torchvision 实现）。"""
import torch.nn as nn

from .._tv import tv_model
from ..registry import register


@register("convnext_tiny")
def convnext_tiny(num_classes, pretrained=False):
    m = tv_model("convnext_tiny", pretrained)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    return m, "classifier"
