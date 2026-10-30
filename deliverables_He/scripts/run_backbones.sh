#!/usr/bin/env bash
# 骨干对比：configs/dogs_cats/backbones/ 下每个模型 × 每个 seed 跑一次，已完成的自动跳过。
# 用法（在 deliverables_He/ 下）：bash scripts/run_backbones.sh [模型名 ...]
#   SEEDS="0 1 2" bash scripts/run_backbones.sh resnet18 vit_b_16
# 环境变量：PY 解释器；SEEDS 种子列表；EXTRA_SET 额外的 --set 覆盖（如 "data.num_workers=4"）；
#           SKIP_SUMMARY=1 跑完不汇总（集群上多个任务并行时由最后统一汇总）
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=${SEEDS:-"0 1"}
PY=${PY:-python}
EXTRA_SET=${EXTRA_SET:-}
mkdir -p runs/backbones
if [ $# -gt 0 ]; then MODELS="$*"; else MODELS=$(ls configs/dogs_cats/backbones | sed -n 's/^b_\(.*\)\.yaml$/\1/p'); fi

for m in $MODELS; do
  for s in $SEEDS; do
    run_id="b_${m}_s${s}"
    out="runs/backbones/${run_id}"
    # history.csv 行数 = 表头 + epochs，跑满才算完成
    epochs=$(sed -n 's/^  epochs: *\([0-9]*\).*/\1/p' "configs/dogs_cats/backbones/b_${m}.yaml")
    if [ -f "$out/history.csv" ] && [ "$(($(wc -l < "$out/history.csv") - 1))" -ge "$epochs" ]; then
      echo "跳过 $run_id（已完成）"; continue
    fi
    echo "=== $run_id ==="
    $PY scripts/train.py --config "configs/dogs_cats/backbones/b_${m}.yaml" --set seed=$s $EXTRA_SET --run-id "$run_id" \
      2>&1 | tee "runs/backbones/${run_id}.log"
  done
done
[ -n "${SKIP_SUMMARY:-}" ] || $PY scripts/summarize_backbones.py
