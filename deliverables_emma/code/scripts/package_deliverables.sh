#!/bin/bash
# 打包 deliverables_emma（可直接拖进 GitHub 网页上传）
# 用法：bash scripts/package_deliverables.sh [目标目录]
set -e
cd "$(dirname "$0")/.."
PROJ="$PWD"
OUT="${1:-/mnt/d/桌面/NTU/6483/project/deliverables_emma}"

rm -rf "$OUT"
mkdir -p "$OUT"/{report,code/scripts,results/{metrics/histories,figures,analysis},docs}

# --- code ---
cp README.md "$OUT/code/README.md"
cp data.py model.py trainer.py train.py predict.py predict_ensemble.py analyze.py cifar10.py imbalance.py figures.py review_submission.py "$OUT/code/"
cp scripts/*.sh "$OUT/code/scripts/"
cp requirements.txt "$OUT/code/"

# --- report ---
cp report/EE6483_Project2_Report_v0.5.docx "$OUT/report/" 2>/dev/null || true
cp report/report_full.md report/report_skeleton.md report/section2_literature.md report/marking_checklist.md "$OUT/report/" 2>/dev/null || true

# --- figures（过滤掉 smoke 之类的临时产物）---
for f in report_figures/*.png; do
  case "$(basename "$f")" in *smoke*) continue;; esac
  cp "$f" "$OUT/results/figures/" 2>/dev/null || true
done

# --- results ---
cp submission.csv submission_ensemble.csv submission_single_resnet18.csv "$OUT/results/" 2>/dev/null || true
cp review/submission_with_probs.csv "$OUT/results/" 2>/dev/null || true
cp analysis/*.png analysis/wrong_samples.csv "$OUT/results/analysis/" 2>/dev/null || true
cp review/audit_100.png "$OUT/results/analysis/" 2>/dev/null || true

# --- metrics（result 每 run 一个；history 合并成一个大文件，控制文件数 <100）---
KEEP_HISTORY="e0_resnet18_160 cifar10 cifar10_pretrained e3_convnext"
python3 - "$OUT" <<'PYEOF'
import glob, json, os, sys
out = sys.argv[1]
base = os.path.join(out, 'results', 'metrics')
keep = {"e0_resnet18_160", "cifar10", "cifar10_pretrained", "e3_convnext"}
all_results, all_hist = {}, {}
for d in sorted(glob.glob('runs/*/')):
    name = os.path.basename(d.rstrip('/'))
    for f in glob.glob(os.path.join(d, '*.json')):
        data = json.load(open(f, encoding='utf-8'))
        if os.path.basename(f) == 'history.json':
            all_hist[name] = data
        else:
            all_results.setdefault(name, {}).update(data)
json.dump(all_results, open(os.path.join(base, 'all_results.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
json.dump(all_hist, open(os.path.join(base, 'all_histories.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
for name in keep:
    if name in all_hist:
        json.dump(all_hist[name], open(os.path.join(base, 'histories', f'{name}.json'), 'w',
                  encoding='utf-8'), ensure_ascii=False, indent=1)
# 混淆矩阵单独留一份（csv 小）
for f in glob.glob('runs/*/*.csv'):
    name = f.split('/')[1]
    os.makedirs(os.path.join(base, 'histories'), exist_ok=True)
    import shutil; shutil.copy(f, os.path.join(base, 'histories', f'{name}_{os.path.basename(f)}'))
PYEOF

cat > "$OUT/README.md" <<'MD'
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
MD

cp docs/reproduce.md docs/experiments.md "$OUT/docs/" 2>/dev/null || true

echo "打包完成: $OUT"
echo "文件数: $(find "$OUT" -type f | wc -l)  大小: $(du -sh "$OUT" | cut -f1)"
find "$OUT" -type f -size +20M | sed 's/^/  [超过 20MB] /'
