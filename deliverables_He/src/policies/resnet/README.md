# resnet/：残差网络（迁移学习主力）

| 注册名 | 参数量 | torchvision DEFAULT 权重 ImageNet top-1 |
|---|---:|---:|
| `resnet18` | 11.7M | 69.8% |
| `resnet34` | 21.8M | 73.3% |
| `resnet50` | 25.6M | 80.9%（IMAGENET1K_V2） |
| `resnext50_32x4d` | 25.0M | 81.2%（IMAGENET1K_V2） |

精度为 torchvision 文档列出的数值，以[官方页面](https://docs.pytorch.org/vision/stable/models.html)为准。

## 结构要点

`conv1 → bn1 → relu → maxpool → layer1 … layer4 → avgpool → fc`。残差连接 `y = F(x) + x` 让梯度可以绕过非线性层传回去，从而能训练很深的网络。分类头参数名前缀为 `fc`。如果设置了 `dropout > 0`，`fc` 会变为 `Dropout + Linear`。

## 两种迁移方式（对应 D2 / D3）

| 设置 | 配置 | 特点 |
|---|---|---|
| 冻结骨干（D2） | `freeze: backbone` | 只训练 `fc`，速度快，不易过拟合。注意：`requires_grad=False` 只冻结参数，train 模式下骨干 BN 的 running mean/var 仍会被新数据更新；如需完全固定，可在训练时把骨干的 BN 层设为 eval 模式 |
| 部分微调 | `freeze: backbone` + `unfreeze_prefixes: [layer4]` | 只调整高层语义特征 |
| 全网络微调（D3） | `freeze: none` + `backbone_lr_mult: 0.1` | 通常精度最高；骨干要用较小学习率，避免破坏预训练特征 |

起始超参数：分类头 `lr=1e-3`，骨干 `1e-4`（mult 0.1），AdamW，`weight_decay=0.01~0.05`，cosine 调度，3~10 个 epoch。猫狗与 ImageNet 的类别重叠较多，通常几个 epoch 就能收敛。

## 参考文献

- He, K., Zhang, X., Ren, S., Sun, J. *Deep Residual Learning for Image Recognition.* CVPR 2016. arXiv:1512.03385
- He, K., Zhang, X., Ren, S., Sun, J. *Identity Mappings in Deep Residual Networks.* ECCV 2016. arXiv:1603.05027
- Xie, S., Girshick, R., Dollár, P., Tu, Z., He, K. *Aggregated Residual Transformations for Deep Neural Networks.* CVPR 2017. arXiv:1611.05431（ResNeXt）
- He, T. et al. *Bag of Tricks for Image Classification with Convolutional Neural Networks.* CVPR 2019. arXiv:1812.01187（warmup、label smoothing、cosine 等训练技巧的对照实验）
- Wightman, R., Touvron, H., Jégou, H. *ResNet strikes back: An improved training procedure in timm.* arXiv:2110.00476, 2021.
- Bello, I. et al. *Revisiting ResNets: Improved Training and Scaling Strategies.* NeurIPS 2021. arXiv:2103.07579
