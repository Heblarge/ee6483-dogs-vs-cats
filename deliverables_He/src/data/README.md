# data/：数据读取与预处理

| 文件 | 内容 |
|---|---|
| `datasets.py` | 读取目录、固定子集抽样、split 清单读写、DataLoader 构建 |
| `transforms.py` | 训练增强与验证/测试的固定预处理（由配置 `data.augment` 控制） |
| `mix.py` | batch 级 Mixup / CutMix（由配置 `train.mix` 控制） |

## 数据约定

```
../datasets/datasets/        （仓库根目录下，相对 deliverables_He/）
├── train/{cat,dog}/*.jpg    各 10,000
├── val/{cat,dog}/*.jpg      各 2,500
└── test/<id>.jpg            500，id 为 1..500
```

- 标签：`cat=0`，`dog=1`（`CLASSES = ("cat", "dog")`）。
- 文件按名称排序，测试集按数字 id 排序，保证每次运行顺序一致。
- 小样本调试：在配置中设 `data.train_per_class: 1000`、`data.val_per_class: 250`，按 `split_seed` 抽样。正式实验建议用 `scripts/make_split.py` 导出 CSV 清单并提交到 `splits/`，配置中写 `data.train_split: splits/xxx.csv`，确保组员使用同一批图像。
- 原图尺寸不一，长宽比也不统一，需要在 transform 中统一尺寸。原数据包含 `.DS_Store`，以及 `__MACOSX/` 下的 `._*` 资源文件，读取时只取图像扩展名，这些文件会被自动跳过。

## 预处理（验证 / 测试，固定）

`Resize(短边 = img_size / 0.875) → CenterCrop(img_size) → ToTensor → Normalize`

- 224 输入时先把短边缩放到 256 再中心裁剪，这是 ImageNet 评估的惯例。
- 使用预训练权重时**必须**用 ImageNet 的 mean/std（`normalize: imagenet`），否则输入分布与预训练时不一致。从零训练的模型也可以沿用这组数值，便于对照。
- 测试时增强（TTA）：`scripts/evaluate.py`、`scripts/predict.py` 加 `--tta`，对原图和水平翻转图的 logits 取平均。

## 训练增强（随机，只用于 train）

| 配置键 | 方法 | 作用与注意事项 |
|---|---|---|
| `random_resized_crop` | 随机面积/比例裁剪后缩放 | 最有效的基础增强。猫狗主体可能偏离中心，`scale` 下限不要太小（建议 ≥0.5），以免裁掉主体 |
| `hflip` | 水平翻转 | 猫狗左右对称，安全；**不要**用垂直翻转 |
| `rotation` | 小角度旋转 | ≤15°，角度过大会引入黑边 |
| `color_jitter` | 亮度/对比度/饱和度/色调 | 模拟光照变化；hue 保持很小，避免毛色失真 |
| `randaugment` | 随机选 N 种操作，幅度 M | 只需调 2 个超参数，适合较大的模型 |
| `trivialaugment` | 每张图随机选 1 种操作和幅度 | 无需调参，效果通常接近 RandAugment |
| `random_erasing` | 随机遮挡矩形区域 | 使模型不只依赖局部特征（如脸部） |
| `train.mix.mixup_alpha` | Mixup：两图线性混合，标签同比例混合 | 正则化，并改善概率校准；训练 loss 会变高，属正常现象 |
| `train.mix.cutmix_alpha` | CutMix：把一图的块贴到另一图上，按面积混合标签 | 对定位和遮挡更鲁棒 |

**建议的增强实验顺序**：D0（无增强）→ D1（RRC + hflip + 轻度 color jitter）→ 有余力再单独加 RandAugment 或 Mixup/CutMix。每次只改一项，才能把精度变化归因到这一项。增强后先用几张样图目视检查，确认主体还在画面里。

## 参考文献

- Shorten, C., Khoshgoftaar, T. M. *A survey on Image Data Augmentation for Deep Learning.* Journal of Big Data, 6:60, 2019.
- Krizhevsky, A., Sutskever, I., Hinton, G. E. *ImageNet Classification with Deep Convolutional Neural Networks.* NeurIPS 2012.（随机裁剪 + 翻转 + PCA 颜色扰动）
- Szegedy, C. et al. *Going Deeper with Convolutions.* CVPR 2015. arXiv:1409.4842（RandomResizedCrop 式的尺度/比例增强）
- Cubuk, E. D. et al. *AutoAugment: Learning Augmentation Strategies from Data.* CVPR 2019. arXiv:1805.09501
- Cubuk, E. D., Zoph, B., Shlens, J., Le, Q. V. *RandAugment: Practical automated data augmentation with a reduced search space.* CVPR Workshops 2020. arXiv:1909.13719
- Müller, S. G., Hutter, F. *TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation.* ICCV 2021. arXiv:2103.10158
- DeVries, T., Taylor, G. W. *Improved Regularization of Convolutional Neural Networks with Cutout.* arXiv:1708.04552, 2017.
- Zhong, Z. et al. *Random Erasing Data Augmentation.* AAAI 2020. arXiv:1708.04896
- Zhang, H., Cisse, M., Dauphin, Y. N., Lopez-Paz, D. *mixup: Beyond Empirical Risk Minimization.* ICLR 2018. arXiv:1710.09412
- Yun, S. et al. *CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features.* ICCV 2019. arXiv:1905.04899
- 数据集来源：Elson, J., Douceur, J. R., Howell, J., Saul, J. *Asirra: A CAPTCHA that Exploits Interest-Aligned Manual Image Categorization.* ACM CCS 2007；Kaggle *Dogs vs. Cats* 竞赛（2013）。本课程数据为其修改版本，以课程划分为准。
- 相关数据集：Parkhi, O. M., Vedaldi, A., Zisserman, A., Jawahar, C. V. *Cats and Dogs.* CVPR 2012.（Oxford-IIIT Pet，可在综述中作为细粒度猫狗分类的参照）
- torchvision 官方：[Transforming and augmenting images](https://docs.pytorch.org/vision/stable/transforms.html)
