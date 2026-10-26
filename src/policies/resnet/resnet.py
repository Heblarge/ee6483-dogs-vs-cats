"""ResNet / ResNeXt 系列（torchvision 实现）。实验 D2（冻结骨干）/ D3（微调）的主力模型。

层名：conv1, bn1, layer1..layer4, fc。微调后部层时设 unfreeze_prefixes: [layer4]。
"""
import torch.nn as nn

from .._tv import tv_model
from ..registry import register

ARCHS = ("resnet18", "resnet34", "resnet50", "resnext50_32x4d")


def _make(arch):
    def build(num_classes, pretrained=False, dropout=0.0):
        m = tv_model(arch, pretrained)
        head = nn.Linear(m.fc.in_features, num_classes)
        m.fc = nn.Sequential(nn.Dropout(dropout), head) if dropout > 0 else head
        return m, "fc"
    return build


for _arch in ARCHS:
    register(_arch)(_make(_arch))
