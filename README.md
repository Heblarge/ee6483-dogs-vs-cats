# EE6483 Mini Project 2：Dogs vs. Cats

本仓库只保留与最终提交有关的文件，结构与提交包一致：

| 路径 | 内容 |
|---|---|
| `EE6483_Group_Report.docx` / `.pdf` | 最终报告（手工维护，以此为准） |
| `submission.csv` | 500 条测试集预测（`id,label`，1 = dog，0 = cat；259 dog / 241 cat） |
| `code/` | 产生报告与 submission 的全部代码、结果与图；说明见 [`code/README.md`](code/README.md) 与 [`code/REPORT_GUIDE.md`](code/REPORT_GUIDE.md) |

提交 NTULearn 的 `EE6483_code.zip` 即 `code/` 目录打包（不含数据与 checkpoint）。

早期的工作副本（`deliverables_emma/`、`PROJECT_OVERVIEW.md`、报告的 Markdown 源文件等）已从当前分支移除，
仍可在 git 历史与 `emma-version` 分支中找到。

## 数据

课程数据不随仓库提交。请把 Dogs vs. Cats 数据解压到仓库根目录：

```text
datasets/datasets/
├── train/{cat,dog}/    # 10,000 + 10,000
├── val/{cat,dog}/      # 2,500 + 2,500
└── test/               # 500 张，1.jpg … 500.jpg
```

`code/` 下的命令默认以 `../datasets/datasets` 作为数据根目录；数据在别处时修改 `--data-root` 即可。
