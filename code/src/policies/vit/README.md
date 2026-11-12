# vit/：Vision Transformer 系列

| 注册名 | 参数量 | 来源 | 注意 |
|---|---:|---|---|
| `vit_b_16` | 86.6M | torchvision | 输入**必须** 224×224；显存占用大 |
| `swin_t` | 28.3M | torchvision | 窗口注意力 + 层级结构，计算量与 ResNet-50 接近 |
| `deit_small` | 22M | timm（`pip install timm`） | 通过 `timm_name` 可换成其他 timm 模型 |

## 原理概要

ViT 把图像切成 16×16 的 patch，线性投影后加上位置编码，再送入标准 Transformer encoder，最后用 `[CLS]` token 分类。它缺少 CNN 的局部性和平移等变性这类归纳偏置，在小数据上**从零训练**效果明显不如 CNN，必须依赖大规模预训练。本项目只做微调对照。

Swin 把注意力限制在局部窗口内，并在相邻层之间平移窗口，同时逐级下采样，得到类似 CNN 的层级特征。它在中等规模数据上更稳定。

## 微调建议

- AdamW，`lr=1e-4~3e-4`（头）+ `backbone_lr_mult=0.1`，`weight_decay=0.05`，warmup 1 个 epoch，cosine 调度。
- 梯度裁剪 `grad_clip_norm=1.0`。Transformer 训练早期梯度容易出现尖峰。
- 增强与正则：RandAugment、Mixup/CutMix、label smoothing（DeiT 训练方案的核心组成）。
- 显存不足时减小 `batch_size`，用 `grad_accum_steps` 补足等效 batch，并开启 `amp`。

配置示例见 `configs/dogs_cats/x_vit_b16_finetune.yaml`。

## 参考文献

- Vaswani, A. et al. *Attention Is All You Need.* NeurIPS 2017. arXiv:1706.03762
- Dosovitskiy, A. et al. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR 2021. arXiv:2010.11929（ViT）
- Touvron, H. et al. *Training data-efficient image transformers & distillation through attention.* ICML 2021. arXiv:2012.12877（DeiT）
- Liu, Z. et al. *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* ICCV 2021. arXiv:2103.14030
- Steiner, A. et al. *How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers.* TMLR 2022. arXiv:2106.10270
- Raghu, M. et al. *Do Vision Transformers See Like Convolutional Neural Networks?* NeurIPS 2021. arXiv:2108.08810（CNN 与 ViT 表征对比，适合写进文献综述）
- Wightman, R. *PyTorch Image Models (timm).* https://github.com/huggingface/pytorch-image-models
