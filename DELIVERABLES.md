# 提交材料（已合并到 deliverables_He/）

原 `deliverables/` 的报告、代码与 submission 已整体移入 [`deliverables_He/`](deliverables_He/)，
在此基础上继续修改。所有命令都在 `deliverables_He/` 下运行。

## 里面有什么

| 位置 | 内容 |
|---|---|
| `deliverables_He/report/EE6483_Group_Report.docx` | **提交用的报告**（A4，10 章 + 参考文献 + 附录） |
| `deliverables_He/report/REPORT.md`、`report/sections/` | 同内容 Markdown（由 sections 自动生成）及其源文件 |
| `deliverables_He/report/backbone_comparison.md` | 骨干对比（7 个模型 × 2 seeds，集群上跑），待并入报告 |
| `deliverables_He/submissions/submission.csv` | **提交用的预测结果**（500 行，261 dog / 239 cat，已通过校验） |
| `deliverables_He/ee6483/` | 报告实验的全部代码、42 次实验的 metrics/日志、图 |
| `deliverables_He/src/`、`configs/`、`cluster/` | 配置驱动的训练框架、骨干对比配置与集群脚本 |
| `deliverables_He/docs/` | 作业 PDF、执行方案 |
| `deliverables_He/REPORT_GUIDE.md` | 报告代码的逐文件说明 + 复现命令 |

## 主要结果

| 模型 | 验证集准确率 |
|---|---|
| HOG + SVM | 75.00% |
| 从零训练的小 CNN | 95.06% |
| ResNet-18 冻结 + 线性头 | 97.78% |
| **ResNet-18 + LP-FT（主模型）** | **98.93 ± 0.05%**（3 seeds） |
| ResNet-18 ×3 seeds + 水平翻转 TTA（提交用） | **99.22%** |
| ConvNeXt-T（骨干对比，统一 6 epoch 协议） | 99.61 ± 0.01%（2 seeds） |

## 怎么跑

```bash
cd deliverables_He
pip install -r requirements.txt

# 数据在仓库根目录 datasets/datasets/{train,val,test}
python -m ee6483.src.data.cache_images --src ../datasets/datasets --dst data/cache --short 256
python -m ee6483.src.train --config ee6483/configs/main.yaml --exp-id E4_r18_ft --seed 42
python -m ee6483.src.predict --ckpts ee6483/results/E4_r18_ft/s42/best.pt --out submissions/submission.csv --tta 1
python ee6483/scripts/validate_submission.py --sub submissions/submission.csv
```

## 说明

- 数据集与模型权重未提交，`.gitignore` 已排除；权重可由上面的命令重新训练得到。
- 报告首页的组员姓名/学号/贡献仍需填写。
