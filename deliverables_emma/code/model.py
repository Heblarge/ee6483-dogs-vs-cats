"""Backbones + classification head.

Supported: resnet18 / resnet50 / convnext_tiny (ImageNet pretrained), and a
CIFAR-friendly ResNet-18 variant (3x3 stem, stride 1, no max-pool) for 32x32
inputs trained from scratch.
"""
import torch.nn as nn
from torchvision import models


def _build_backbone(arch, pretrained):
    if arch == 'resnet18':
        w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet18(weights=w)
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
    elif arch == 'resnet50':
        w = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = models.resnet50(weights=w)
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
    elif arch == 'convnext_tiny':
        w = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.convnext_tiny(weights=w)
        feat_dim = m.classifier[-1].in_features
        m.classifier = nn.Sequential(m.classifier[0], nn.Flatten(1))
    elif arch == 'resnet18_cifar':
        # CIFAR stem: 3x3 stride-1 conv, no max-pool -> 32x32 keeps spatial size
        m = models.resnet18(weights=None, num_classes=10)
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
    else:
        raise ValueError(f'Unknown arch: {arch}')
    return m, feat_dim


class Classifier(nn.Module):
    def __init__(self, arch='resnet18', num_classes=1, pretrained=True, dropout=0.2):
        super().__init__()
        self.backbone, feat_dim = _build_backbone(arch, pretrained)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, num_classes))

    def forward(self, x):
        feat = self.backbone(x)
        if feat.dim() > 2:
            feat = feat.flatten(1)
        return self.head(feat)

    def set_backbone_trainable(self, flag: bool):
        for p in self.backbone.parameters():
            p.requires_grad = flag

    def backbone_parameters(self):
        return self.backbone.parameters()

    def head_parameters(self):
        return self.head.parameters()
