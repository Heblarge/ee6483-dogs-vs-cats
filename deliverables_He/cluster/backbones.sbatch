#!/bin/bash
# 骨干对比：job array，每个任务 1 块 GPU 依次跑一组模型的全部 seed（默认 0 1）。
# msc 账户每人最多提交 5 个、同时运行 2 个任务，所以 7 个模型分成 2 组，计算量大致均衡。
# 在 deliverables_He/ 下提交：sbatch cluster/backbones.sbatch
#   换 GPU 型号：sbatch --gres=gpu:l40:1 cluster/backbones.sbatch
#   只补跑某一组：sbatch --array=1 cluster/backbones.sbatch   （下标见 MODEL_SETS）
# 已跑完的 run 自动跳过，任务被中断后重新提交即可。全部结束后在登录节点外汇总（见 cluster/README.md）。
#SBATCH --job-name=ee6483-bb
#SBATCH --gres=gpu:a40:1
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=cluster/logs/%x-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh

MODEL_SETS=("vit_b_16 resnet50 resnet18 efficientnet_b0" "swin_t convnext_tiny resnet34")
M=${MODEL_SETS[$SLURM_ARRAY_TASK_ID]}

module load Miniforge3
set +u; source activate "$ENV"; set -u   # conda 的激活脚本不兼容 set -u
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "models=$M seeds=${SEEDS:-0 1} node=$(hostname)"

# 每块 GPU 只配 4 个 CPU，DataLoader worker 相应设为 4；多个任务并行，最后统一汇总
EXTRA_SET="data.num_workers=4" SKIP_SUMMARY=1 PY=python bash scripts/run_backbones.sh $M
