# 已完成的工作（deliverables/）

`deliverables_He/` 下的 `src/`、`scripts/`、`configs/` 是**项目骨架与实施计划**（见 `PROJECT_OVERVIEW.md`），
尚未包含实现。已完成的一整套工作放在 [`deliverables/`](deliverables/)，自成一体、可直接复现。

## 里面有什么

| 位置 | 内容 |
|---|---|
| `deliverables/report/EE6483_Group_Report.docx` | **提交用的报告**（A4，10 章 + 参考文献 + 附录，18 图 43 表） |
| `deliverables/report/REPORT.md` | 同内容 Markdown（自动生成） |
| `deliverables/submission.csv` | **提交用的预测结果**（500 行，261 dog / 239 cat，已通过校验） |
| `deliverables/ee6483/` | 全部代码（27 个模块）、42 次实验的 metrics/日志、24 张图 |
| `deliverables/docs/` | 作业 PDF、执行方案与"计划 vs 实际"对照 |
| `deliverables/README.md` | 逐文件说明 + 复现命令 |

## 主要结果

| 模型 | 验证集准确率 |
|---|---|
| HOG + SVM | 75.00% |
| 从零训练的小 CNN | 95.06% |
| ResNet-18 冻结 + 线性头 | 97.78% |
| **ResNet-18 + LP-FT（主模型）** | **98.93 ± 0.05%**（3 seeds） |
| ResNet-18 ×3 seeds + 水平翻转 TTA（提交用） | **99.22%** |

## 怎么跑

```bash
cd deliverables
pip install -r requirements.txt

# 数据放到 data/datasets/{train,val,test}（见 README 的 Data 一节），然后：
python -m ee6483.src.data.cache_images --src data/datasets --dst data/cache --short 256
python -m ee6483.src.train --config ee6483/configs/main.yaml --exp-id E4_r18_ft --seed 42
python -m ee6483.src.predict --ckpts ee6483/results/E4_r18_ft/s42/best.pt --out submission.csv --tta 1
python ee6483/scripts/validate_submission.py --sub submission.csv
```

## 说明

- 数据集（3 GB）与模型权重（1.8 GB）未提交，`.gitignore` 已排除；权重可由上面的命令重新训练得到。
- 报告首页的组员姓名/学号三行仍需填写。
- 本目录不改动 `deliverables_He/` 的骨架代码，两者可以并存；如需合并成一套结构，可另行整理。
