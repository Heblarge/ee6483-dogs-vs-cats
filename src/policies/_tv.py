"""torchvision 预训练权重的统一加载。pretrained=True 使用该模型的 DEFAULT 权重。"""
from torchvision import models


def tv_model(arch, pretrained):
    weights = models.get_model_weights(arch).DEFAULT if pretrained else None
    return models.get_model(arch, weights=weights)
