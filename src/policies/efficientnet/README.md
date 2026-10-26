# efficientnet/：高效卷积网络

| 注册名 | 参数量 | torchvision DEFAULT 权重 ImageNet top-1 | 说明 |
|---|---:|---:|---|
| `efficientnet_b0` | 5.3M | 77.7% | 精度与参数量平衡最好的入门选择 |
| `efficientnet_v2_s` | 21.5M | 84.2% | 官方权重评估时用 384 输入；本项目统一 224 时精度会低一些，需在报告中说明 |
| `mobilenet_v3_large` | 5.5M | 75.3% | 速度最快，适合 CPU 或笔记本 GPU 快速试验 |

精度以 [torchvision 文档](https://docs.pytorch.org/vision/stable/models.html)为准。分类头前缀均为 `classifier`。MobileNetV3 的 `classifier` 是 2 层 MLP（约 1.2M 参数），冻结骨干时这两层都会训练。

## 原理概要

- **MobileNetV2/V3**：用深度可分离卷积和倒残差块（inverted residual，先升维再降维）大幅减少计算量；V3 加入 SE 注意力和 h-swish，并通过 NAS 搜索结构。
- **EfficientNet**：在 MBConv 基础上提出“复合缩放”，按固定比例同时放大深度、宽度和输入分辨率。B0 是 NAS 搜出的基础网络。
- **EfficientNetV2**：浅层改用 Fused-MBConv 加快训练，并配合渐进式调整图像尺寸。

## 使用建议

- 这一族模型的 BN 和 SE 层对小 batch 比较敏感，batch 尽量不小于 32。
- 微调设置与 ResNet 相同即可（见 `configs/dogs_cats/x_efficientnet_b0_finetune.yaml`）。
- 在报告中可以把“参数量 / 推理速度 vs 验证精度”画成散点图，用于讨论模型选择。

## 参考文献

- Howard, A. G. et al. *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.* arXiv:1704.04861, 2017.
- Sandler, M. et al. *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR 2018. arXiv:1801.04381
- Howard, A. et al. *Searching for MobileNetV3.* ICCV 2019. arXiv:1905.02244
- Hu, J., Shen, L., Sun, G. *Squeeze-and-Excitation Networks.* CVPR 2018. arXiv:1709.01507
- Tan, M., Le, Q. V. *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML 2019. arXiv:1905.11946
- Tan, M., Le, Q. V. *EfficientNetV2: Smaller Models and Faster Training.* ICML 2021. arXiv:2104.00298
