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

## 骨干对比（`dogs_cats/backbones/`）

回答 f)"不同模型对验证准确率的影响"。7 个文件只有 `model.name`、`data.batch_size`、`train.grad_accum_steps` 不同，其余完全一致：
224px、ImageNet 预训练、全网络微调（骨干 lr 1e-4 / 头 1e-3）、AdamW、cosine + 1 epoch warmup、label smoothing 0.1、
增强只用 RandomResizedCrop + 水平翻转、**固定 6 个 epoch、不早停**、有效 batch 64。

| 文件 | 模型 | 参数量 | 类别 |
|---|---|---:|---|
| `b_efficientnet_b0.yaml` | EfficientNet-B0 | 4.0M | 轻量 CNN |
| `b_resnet18.yaml` | ResNet-18 | 11.2M | 经典 CNN |
| `b_resnet34.yaml` | ResNet-34 | 21.3M | 经典 CNN |
| `b_resnet50.yaml` | ResNet-50 | 23.5M | 经典 CNN（瓶颈块） |
| `b_swin_t.yaml` | Swin-T | 27.5M | 层级 Transformer |
| `b_convnext_tiny.yaml` | ConvNeXt-T | 27.8M | 现代化 CNN |
| `b_vit_b_16.yaml` | ViT-B/16 | 85.8M | 纯 Transformer（batch 32 × 累积 2） |

```bash
bash scripts/run_backbones.sh                         # 全部模型 × seed 0,1，已完成的跳过；结束后自动汇总
SEEDS="0 1 2" bash scripts/run_backbones.sh resnet34  # 只跑指定模型 / 加 seed
python scripts/summarize_backbones.py                 # 单独汇总 -> results/backbones.{csv,md}
```

汇总表同时给出 best（按验证集挑的 epoch）和 last（第 6 个 epoch）准确率，以及参数量和每 epoch 耗时。
注意：各模型的 torchvision 预训练权重配方不同（如 ResNet-50 用 IMAGENET1K_V2），写报告时需说明这是"模型 + 其预训练权重"的整体对比。

## 调试

快速跑通流程（每类 200 张、1 个 epoch）：

```bash
python scripts/train.py --config configs/dogs_cats/d0_simple_cnn.yaml \
  --set data.train_per_class=200 data.val_per_class=50 train.epochs=1 data.num_workers=0
```

CIFAR-10 配置之后放在 `configs/cifar10/`。
