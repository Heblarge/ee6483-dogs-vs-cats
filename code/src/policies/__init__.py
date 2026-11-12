"""导入各模型族以完成注册。新增模型族时在这里加一行 import。"""
from . import cnn, convnext, efficientnet, resnet, vit  # noqa: F401
from .registry import MODEL_REGISTRY, build_model, register
