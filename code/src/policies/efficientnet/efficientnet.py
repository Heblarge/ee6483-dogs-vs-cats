"""高效 CNN：EfficientNet（复合缩放）与 MobileNetV3（移动端轻量）。算力有限时的首选对照。"""
import torch.nn as nn

from .._tv import tv_model
from ..registry import register


def _replace_last_linear(m, index, num_classes):
    m.classifier[index] = nn.Linear(m.classifier[index].in_features, num_classes)
    return m, "classifier"


@register("efficientnet_b0")
def efficientnet_b0(num_classes, pretrained=False):
    return _replace_last_linear(tv_model("efficientnet_b0", pretrained), 1, num_classes)


@register("efficientnet_v2_s")
def efficientnet_v2_s(num_classes, pretrained=False):
    return _replace_last_linear(tv_model("efficientnet_v2_s", pretrained), 1, num_classes)


@register("mobilenet_v3_large")
def mobilenet_v3_large(num_classes, pretrained=False):
    return _replace_last_linear(tv_model("mobilenet_v3_large", pretrained), 3, num_classes)
