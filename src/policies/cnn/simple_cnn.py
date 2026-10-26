"""从零训练的简单 CNN 基线（实验 D0/D1）。

结构：N 个 [Conv-BN-ReLU ×2 + MaxPool] 块 -> 全局平均池化 -> Dropout -> Linear。
输入 [B, 3, H, W]，输出 [B, num_classes] logits；任意输入尺寸都可用（GAP 去掉了尺寸依赖）。
"""
import torch.nn as nn

from ..registry import register


def conv_bn_relu(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2, channels=(32, 64, 128, 256), dropout=0.3):
        super().__init__()
        blocks, cin = [], 3
        for cout in channels:
            blocks += [conv_bn_relu(cin, cout), conv_bn_relu(cout, cout), nn.MaxPool2d(2)]
            cin = cout
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(cin, num_classes))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


@register("simple_cnn")
def simple_cnn(num_classes, pretrained=False, channels=(32, 64, 128, 256), dropout=0.3):
    if pretrained:
        raise ValueError("simple_cnn 没有预训练权重")
    return SimpleCNN(num_classes, tuple(channels), dropout), "classifier"
