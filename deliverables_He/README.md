# EE6483 Mini Project 2：Dogs vs. Cats — He 的版本

项目目标、分工与进度见仓库根目录的 [PROJECT_OVERVIEW.md](../PROJECT_OVERVIEW.md)。本文件说明代码结构与运行方式。

本目录包含两套相互独立的代码：

- **`ee6483/` + `report/` + `submissions/`**：最终报告、提交用的 `submission.csv` 及产生它们的 42 次实验（原 `deliverables/`，见 [REPORT_GUIDE.md](REPORT_GUIDE.md)）。
- **`src/` + `scripts/` + `configs/` + `cluster/`**：配置驱动的训练框架，用于骨干对比（`report/backbone_comparison.md`）及集群实验。下文说明这一套。

> 本目录自成一体，**所有命令都在 `deliverables_He/` 下运行**。数据集仍放在仓库根目录的 `datasets/`（与其他成员的版本共用），配置中以 `../datasets/datasets` 引用。

## 目录结构

```
deliverables_He/
├── ee6483/                  报告实验的代码、结果与图（原 deliverables/）  → REPORT_GUIDE.md
├── configs/                  实验配置（YAML，每个文件 = 一个实验）  → configs/README.md
│   └── dogs_cats/            D0~D3 及扩展对照
│       └── backbones/        骨干对比：7 个模型，统一协议
├── src/
│   ├── data/                 数据读取、预处理、增强、Mixup/CutMix   → 预处理方法与引用
│   ├── policies/             模型库（注册表 + 各模型族）            → 选型建议与引用
│   │   ├── cnn/              SimpleCNN、VGG
│   │   ├── resnet/           ResNet-18/34/50、ResNeXt-50
│   │   ├── efficientnet/     EfficientNet-B0/V2-S、MobileNetV3
│   │   ├── convnext/         ConvNeXt-T
│   │   └── vit/              ViT-B/16、Swin-T、DeiT-S
│   ├── training/             forward/backward、梯度裁剪、优化器、调度器、损失  → 训练原理与引用
│   ├── evaluation/           指标、验证评估、测试预测与 submission 导出         → 评估约定与引用
│   └── utils/                配置读取、随机种子
├── scripts/                  命令行入口：train / evaluate / predict / make_split；run_backbones.sh / summarize_backbones.py
├── cluster/                  NTU EEE GPU 集群脚本（sync / 环境 / sbatch）  → CLUSTER_GUIDE.md
├── splits/                   固定样本清单 CSV（提交 Git，不含图像）
├── docs/                     作业 PDF、执行方案
├── results/                  用于报告的精简指标表与图
├── submissions/              提交用的 submission.csv（及格式模板、概率）
├── report/                   报告（.docx、REPORT.md、sections/ 源文件与构建脚本；backbone_comparison.md：骨干对比章节）
└── runs/                     训练输出：config、history.csv、checkpoints（不提交 Git）
```

每个 `src/` 子目录都有 README.md，记录该部分的方法说明、可选项和参考文献，写报告时可直接引用。

## 环境

```bash
python -m venv .venv && source .venv/bin/activate
# 先按 https://pytorch.org/get-started/locally/ 安装与 CUDA 匹配的 torch / torchvision
pip install -r requirements.txt
```

## 数据

把课程数据解压到仓库根目录的 `datasets/datasets/`（即本目录的 `../datasets/datasets/`），目录应为 `train/{cat,dog}`、`val/{cat,dog}`、`test/`，各含 10000×2、2500×2、500 张图像。路径可通过配置项 `data.root` 修改。

## 运行

```bash
# 1. 快速跑通（小样本、1 epoch）
python scripts/train.py --config configs/dogs_cats/d0_simple_cnn.yaml \
  --set data.train_per_class=200 data.val_per_class=50 train.epochs=1 data.num_workers=0

# 2. 正式训练
python scripts/train.py --config configs/dogs_cats/d3_resnet18_finetune.yaml

# 3. 验证集评估 + 误分类清单
python scripts/evaluate.py --ckpt runs/<run_id>/checkpoints/best.pt

# 4. 生成提交文件
python scripts/predict.py --ckpt runs/<run_id>/checkpoints/best.pt --out submissions/submission.csv

# 5. 骨干对比（7 个模型 × 2 seeds，跑完自动汇总到 results/backbones.md）
bash scripts/run_backbones.sh
python scripts/plot_backbones.py runs/backbones results/backbones.png   # 画图，需 matplotlib
```

在 NTU EEE GPU 集群上运行：见 [CLUSTER_GUIDE.md](CLUSTER_GUIDE.md)。

## 接口约定

- 标签 `cat=0`，`dog=1`。
- 模型输入 `[B, 3, H, W]`，输出 `[B, C]` logits；损失函数内部做 softmax，模型输出不要提前做。
- 所有超参数都写在配置里，不在代码中写死路径或参数。
- 新增模型：在 `src/policies/<family>/` 中用 `@register("name")` 注册（见 `src/policies/README.md`）。
