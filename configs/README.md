# configs/：实验配置

每个 YAML 是一个完整的实验，不依赖继承，读一个文件就能知道全部设置。命令行可临时覆盖：

```bash
python scripts/train.py --config configs/dogs_cats/d3_resnet18_finetune.yaml --set optim.lr=3e-4 seed=1
```

被覆盖后的最终配置会保存到 `runs/<run_id>/config.yaml`，同时写入 checkpoint。

| 文件 | 实验 | 与谁对照 | 改变的变量 |
|---|---|---|---|
| `dogs_cats/d0_simple_cnn.yaml` | D0 简单 CNN，无增强 | — | 基线 |
| `dogs_cats/d1_simple_cnn_aug.yaml` | D1 简单 CNN + 增强 | D0 | 只有 `data.augment` |
| `dogs_cats/d2_resnet18_frozen.yaml` | D2 ResNet-18 冻结骨干 | — | 预训练特征 + 线性分类头 |
| `dogs_cats/d3_resnet18_finetune.yaml` | D3 ResNet-18 全网络微调 | D2 | `freeze`、学习率、label smoothing、梯度裁剪 |
| `dogs_cats/x_efficientnet_b0_finetune.yaml` | 扩展：EfficientNet-B0 微调 | D3 | 骨干网络 |
| `dogs_cats/x_vit_b16_finetune.yaml` | 扩展：ViT-B/16 微调 | D3 | 骨干网络（Transformer） |

注意：D3 相对 D2 改动了多个变量（为了得到合理的微调配置）。如果报告中要严格归因，增加一组只改 `freeze` 的对照：`--set model.freeze=none optim.backbone_lr_mult=0.1`。

## 调试

快速跑通流程（每类 200 张、1 个 epoch）：

```bash
python scripts/train.py --config configs/dogs_cats/d0_simple_cnn.yaml \
  --set data.train_per_class=200 data.val_per_class=50 train.epochs=1 data.num_workers=0
```

CIFAR-10 配置之后放在 `configs/cifar10/`。
