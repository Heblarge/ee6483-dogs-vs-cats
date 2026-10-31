#!/bin/bash
# 骨干对比：job array，每个任务 1 块 GPU 跑一个模型的全部 seed（默认 0 1）。
# 在 deliverables_He/ 下提交：sbatch cluster/backbones.sbatch
#   换 GPU 型号：sbatch --gres=gpu:l40:1 cluster/backbones.sbatch
#   只补跑某几个：sbatch --array=2,6 cluster/backbones.sbatch   （下标见 MODELS）
# 已跑完的 run 自动跳过，任务被中断后重新提交即可。全部结束后在登录节点外汇总（见 cluster/README.md）。
#SBATCH --job-name=ee6483-bb
#SBATCH --gres=gpu:a40:1
#SBATCH --time=6:00:00
#SBATCH --array=0-6
#SBATCH --output=cluster/logs/%x-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh

MODELS=(resnet18 resnet34 resnet50 efficientnet_b0 convnext_tiny swin_t vit_b_16)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}

module load Miniforge3
set +u; source activate "$ENV"; set -u   # conda 的激活脚本不兼容 set -u
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "model=$M seeds=${SEEDS:-0 1} node=$(hostname)"

# 每块 GPU 只配 4 个 CPU，DataLoader worker 相应设为 4；多个任务并行，最后统一汇总
EXTRA_SET="data.num_workers=4" SKIP_SUMMARY=1 PY=python bash scripts/run_backbones.sh "$M"
