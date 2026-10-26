# convnext/：现代纯卷积网络

| 注册名 | 参数量 | torchvision DEFAULT 权重 ImageNet top-1 |
|---|---:|---:|
| `convnext_tiny` | 28.6M | 82.5% |

分类头前缀为 `classifier`（LayerNorm → Flatten → Linear，只替换最后的 Linear）。

## 原理概要

ConvNeXt 以 ResNet-50 为起点，逐步引入 Swin Transformer 的设计：分阶段计算比例（3:3:9:3）、patchify stem（4×4 stride 4）、7×7 depthwise 卷积、倒瓶颈、LayerNorm 替代 BN、GELU、更少的激活与归一化层。结果是在同等计算量下与 Swin 持平或更好，同时保留了卷积的简单和高效。

它很适合放在报告中**连接 CNN 与 ViT**：两者的差异有多少来自架构本身，又有多少来自训练方案和设计细节？

## 使用建议

- 微调设置：AdamW，`lr=1e-4~5e-4`，`backbone_lr_mult=0.1`，`weight_decay=0.05`，配合 label smoothing。
- 原论文训练使用了 stochastic depth（drop path）；torchvision 的 `convnext_tiny` 可通过 `stochastic_depth_prob` 设置，如需要可在构建函数中加入。

## 参考文献

- Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., Xie, S. *A ConvNet for the 2020s.* CVPR 2022. arXiv:2201.03545
- Woo, S. et al. *ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders.* CVPR 2023. arXiv:2301.00808
- Huang, G. et al. *Deep Networks with Stochastic Depth.* ECCV 2016. arXiv:1603.09382
