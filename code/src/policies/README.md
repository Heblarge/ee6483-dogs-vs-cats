# policies/：模型库

每个子目录是一个模型族，通过 `@register("name")` 注册到 `registry.py`，训练脚本只按配置里的 `model.name` 取模型。所有模型的接口一致：

- 输入 `[B, 3, H, W]`，输出 `[B, num_classes]` logits，不经过 softmax。
- 构建函数返回 `(model, head_name)`。`head_name` 是分类头参数名前缀，用于两件事：`freeze: backbone` 时只训练分类头；`optim.backbone_lr_mult` 让骨干使用更小的学习率。

| 子目录 | 注册名 | 参数量 | 适用实验 | 说明 |
|---|---|---:|---|---|
| [cnn/](cnn/) | `simple_cnn`、`vgg11_bn`、`vgg16_bn` | 1.2M / 133M / 138M | D0、D1 | 从零训练的基线；VGG 作为经典 CNN 对照 |
| [resnet/](resnet/) | `resnet18`、`resnet34`、`resnet50`、`resnext50_32x4d` | 11.7M–25.6M | D2、D3 | 迁移学习主力，残差连接 |
| [efficientnet/](efficientnet/) | `efficientnet_b0`、`efficientnet_v2_s`、`mobilenet_v3_large` | 5.3M–21.5M | 扩展对照 | 参数少、速度快，适合算力有限时 |
| [convnext/](convnext/) | `convnext_tiny` | 28.6M | 扩展对照 | 吸收 Transformer 设计经验的现代纯卷积网络 |
| [vit/](vit/) | `vit_b_16`、`swin_t`、`deit_small` | 86.6M / 28.3M / 22M | 扩展对照 | 自注意力模型；`deit_small` 需要安装 timm |

参数量为 ImageNet 1000 类分类头下的数值，来自 torchvision 文档与各原论文；换成 2 类头后略少。`simple_cnn` 的数值按默认 `channels=[32,64,128,256]` 计算。

## 选型建议

1. **必做**：`simple_cnn`（D0/D1）和 `resnet18`（D2/D3）。这两组对应项目计划里的核心对照。
2. **推荐加一个**：`efficientnet_b0` 或 `convnext_tiny`，用于讨论模型规模和架构对迁移效果的影响。
3. **可选**：`vit_b_16` / `swin_t`，用于在文献综述中对比 CNN 和 Transformer 的归纳偏置。ViT 在小数据上从零训练效果很差，应只做微调。
4. 若要单独说明“预训练本身”的贡献，增加 `resnet18` + `pretrained: false` 的对照，与 D2/D3 使用相同的结构。

## 新增模型

在合适的子目录（或新建子目录）写构建函数：

```python
from ..registry import register

@register("my_net")
def my_net(num_classes, pretrained=False, **kwargs):
    model = ...
    return model, "classifier"   # 分类头参数名前缀
```

新建子目录时，在 `policies/__init__.py` 中 import 它。

## 迁移学习通用参考

- Yosinski, J. et al. *How transferable are features in deep neural networks?* NeurIPS 2014. arXiv:1411.1792
- Kornblith, S., Shlens, J., Le, Q. V. *Do Better ImageNet Models Transfer Better?* CVPR 2019. arXiv:1805.08974
- Kolesnikov, A. et al. *Big Transfer (BiT): General Visual Representation Learning.* ECCV 2020. arXiv:1912.11370
- Deng, J. et al. *ImageNet: A Large-Scale Hierarchical Image Database.* CVPR 2009.
- PyTorch 官方：[Transfer Learning for Computer Vision Tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- torchvision 官方：[Models and pre-trained weights](https://docs.pytorch.org/vision/stable/models.html)（各权重的精度、参数量与配套预处理）
