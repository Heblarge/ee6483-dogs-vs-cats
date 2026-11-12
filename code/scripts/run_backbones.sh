#!/usr/bin/env bash
# 骨干对比：configs/dogs_cats/backbones/ 下每个模型 × 每个 seed 跑一次，已完成的自动跳过。
# 用法（在 code/ 下）：bash scripts/run_backbones.sh [模型名 ...]
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
    # history.csv 行数 = 表头 + epochs；同时必须有可用于评估/集成的 best.pt。
    # history.csv 会提交到 git，而 checkpoint 不会，因此只检查 history 会让全新
    # clone 错误地跳过训练，随后在集成阶段失败。
    epochs=$(sed -n 's/^  epochs: *\([0-9]*\).*/\1/p' "configs/dogs_cats/backbones/b_${m}.yaml")
    if [ -z "$epochs" ]; then
      echo "错误：无法从 configs/dogs_cats/backbones/b_${m}.yaml 读取 train.epochs" >&2
      exit 2
    fi
    history_epochs=0
    if [ -f "$out/history.csv" ]; then
      history_epochs=$(($(wc -l < "$out/history.csv") - 1))
    fi
    ckpt="$out/checkpoints/best.pt"
    if [ "$history_epochs" -ge "$epochs" ] && [ -s "$ckpt" ]; then
      echo "跳过 $run_id（history 和 checkpoint 均完整）"; continue
    fi
    if [ "$history_epochs" -ge "$epochs" ] && [ ! -s "$ckpt" ]; then
      echo "重跑 $run_id（history 已存在，但缺少 checkpoint）"
    fi
    echo "=== $run_id ==="
    $PY scripts/train.py --config "configs/dogs_cats/backbones/b_${m}.yaml" --set seed=$s $EXTRA_SET --run-id "$run_id" \
      2>&1 | tee "runs/backbones/${run_id}.log"
  done
done
[ -n "${SKIP_SUMMARY:-}" ] || $PY scripts/summarize_backbones.py
