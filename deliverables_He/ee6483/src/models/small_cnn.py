"""A compact CNN trained from scratch (no ImageNet weights).

Purpose: establish what is achievable *without* pretraining, so the report can
separate "the model is good" from "the pretrained features already know cats
and dogs". Roughly 2.3M parameters.
"""
from __future__ import annotations

import torch.nn as nn


def conv_bn_relu(c_in: int, c_out: int, k: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, k, padding=k // 2, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 2, width: tuple[int, ...] = (32, 64, 128, 256), p_drop: float = 0.3):
        super().__init__()
        blocks = []
        c_in = 3
        for c in width:
            blocks += [conv_bn_relu(c_in, c), conv_bn_relu(c, c), nn.MaxPool2d(2)]
            c_in = c
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(p_drop), nn.Linear(c_in, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))
