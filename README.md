# EE6483 Mini Project 2：Dogs vs. Cats

> **最终提交物（以此为准）**
> - 报告：`deliverables_He/report/EE6483_Group_Report.docx`
> - 预测：`deliverables_He/submissions/submission.csv`（500 行，259 dog / 241 cat）
>
> `deliverables_emma/` 是组员的工作副本，**不是**本次提交版本；其报告中的 CIFAR/不平衡数字与最终报告不同，请勿混用。

项目目标与分工见 [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)，作业要求见 [EE6483-Project2.pdf](EE6483-Project2.pdf)。

各成员的版本各自独立成目录，互不依赖：

| 目录 | 作者 | 说明 |
|---|---|---|
| [`deliverables_He/`](deliverables_He/) | He（`ee6483/`、`report/` 源自 Darling-2） | **最终报告与 submission**、42 次实验（`ee6483/`）；配置驱动的训练框架与骨干对比（`src/`、`cluster/`）。见 [DELIVERABLES.md](DELIVERABLES.md) |
| [`deliverables_emma/`](deliverables_emma/) | Emma | 工作副本：DvC + CIFAR-10 + 类别不平衡。**不用于提交** |

## 数据

课程数据不随仓库提交。请把课程提供的 Dogs vs. Cats 数据解压到仓库根目录：

```text
datasets/datasets/
├── train/{cat,dog}/    # 10,000 + 10,000
├── val/{cat,dog}/      # 2,500 + 2,500
└── test/               # 500 张，1.jpg … 500.jpg
```

`deliverables_He/` 下的命令默认以 `../datasets/datasets` 作为数据根目录；如果你的数据在别处，把命令中的 `--data-root` 改成实际路径即可。
