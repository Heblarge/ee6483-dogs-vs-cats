# EE6483 Project 2 — Emma 的交付版本

Dogs vs. Cats 二分类 + CIFAR-10 十分类 + 类别不平衡实验。
本目录自成一体：代码、结果、图表、报告齐全；不含模型权重与数据集（见"如何复现"）。

## 关键结果

| 任务 | 方法 | 验证集 | 测试集 | 耗时 |
|---|---|---|---|---|
| Dogs vs Cats | ResNet-18 @160px，15 epoch，两阶段微调 | **98.80%** | —（test 无标签） | 67 min |
| CIFAR-10 | CIFAR stem ResNet-18 @32px，从零训练 20 epoch | 87.92% | **88.10%** | 106 min |
| CIFAR-10 | ImageNet 预训练 ResNet-18 @64px，15 epoch | 93.52% | **92.96%** | 25 min |
| 类别不平衡 | baseline / 过采样 / 加权 CE / focal | 见 results/metrics/ | — | — |

DvC 补充：Precision 0.9884 / Recall 0.9876 / F1 0.9880；5000 张验证集错 60 张（29 猫→狗、31 狗→猫）。

**测试集人工审核**：随机抽 100 张逐张肉眼核对，**100 张全部正确**；0/100 在 95% 置信下错误率上界 2.95%（测试准确率 ≥97%）。
审核图：`results/analysis/audit_100.png`。

## 目录

```
deliverables_emma/
├── report/      报告 docx + Markdown 源 + 评分自查表 + 图
├── code/        数据/模型/训练/预测/分析/消融 + 脚本 + requirements
├── results/     submission.csv、各实验 metrics、图、错误分析
└── docs/        复现指南、实验清单（实验→报告章节对照）
```

## 复现

```bash
pip install -r code/requirements.txt
bash code/scripts/run_baseline.sh        # DvC 主线：训练→submission→分析→出图（约 90 分钟）
bash code/scripts/run_cifar.sh           # CIFAR-10 + 不平衡实验
bash code/scripts/run_queue.sh           # 全部对照实验（已有结果自动跳过）
```

数据目录约定见 `docs/复现指南.md`。

## 设计要点（评审关注）

1. **无测试集泄漏**：CIFAR-10 从训练集切 5,000 张做验证，官方 test 只在最后评估一次。
2. **错误分析双层**：验证集 5,000 张定量分析 + 测试集 100 张人工审核。
3. **两阶段微调**：先冻结 backbone 训练分类头，再全网络微调（backbone 学习率 0.1×）。
4. **可复现**：固定 seed（含 DataLoader worker），完整参数记录在 `result.json`。
5. **诚实的负结果**：水平翻转 TTA 在 500 张测试图上未改变任何预测（概率饱和），报告中如实说明。

## 环境

WSL2 Ubuntu / Python 3.12 / PyTorch 2.14.1 + CUDA 12.6 / 单卡 MX450（2GB），全程本地训练。
