"""Vision Transformer 系列。

- vit_b_16：torchvision 实现，输入必须是 224×224（位置编码固定）。
- swin_t：层级式窗口注意力 Transformer，torchvision 实现。
- deit_small：需要 `pip install timm`，数据高效的 ViT 训练方案。
"""
import torch.nn as nn

from .._tv import tv_model
from ..registry import register


@register("vit_b_16")
def vit_b_16(num_classes, pretrained=False):
    m = tv_model("vit_b_16", pretrained)
    m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
    return m, "heads"


@register("swin_t")
def swin_t(num_classes, pretrained=False):
    m = tv_model("swin_t", pretrained)
    m.head = nn.Linear(m.head.in_features, num_classes)
    return m, "head"


@register("deit_small")
def deit_small(num_classes, pretrained=False, timm_name="deit_small_patch16_224"):
    import timm  # 可选依赖

    return timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes), "head"
