"""VGG（torchvision 实现，带 BN）。参数量大（vgg16_bn ≈ 138M），主要用作经典 CNN 对照。"""
import torch.nn as nn

from .._tv import tv_model
from ..registry import register


def _vgg(arch, num_classes, pretrained):
    m = tv_model(arch, pretrained)
    m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)
    return m, "classifier"


@register("vgg11_bn")
def vgg11_bn(num_classes, pretrained=False):
    return _vgg("vgg11_bn", num_classes, pretrained)


@register("vgg16_bn")
def vgg16_bn(num_classes, pretrained=False):
    return _vgg("vgg16_bn", num_classes, pretrained)
