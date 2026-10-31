#!/bin/bash
# 单个实验：任意配置 + 任意 --set 覆盖，跑在 1 块 GPU 上。
# 在 deliverables_He/ 下提交：
#   sbatch cluster/train.sbatch configs/dogs_cats/d3_resnet18_finetune.yaml
#   sbatch cluster/train.sbatch configs/dogs_cats/d3_resnet18_finetune.yaml --set seed=1 optim.lr=3e-4
#   sbatch --gres=gpu:l40:1 --time=2:00:00 cluster/train.sbatch <config>
#SBATCH --job-name=ee6483-train
#SBATCH --gres=gpu:a40:1
#SBATCH --time=4:00:00
#SBATCH --output=cluster/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh
CFG=${1:?用法: sbatch cluster/train.sbatch <config.yaml> [--set k=v ...]}
shift

module load Miniforge3
set +u; source activate "$ENV"; set -u   # conda 的激活脚本不兼容 set -u
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 每块 GPU 只配 4 个 CPU，DataLoader worker 设为 4；后面的 --set 可再覆盖
python scripts/train.py --config "$CFG" --set data.num_workers=4 "$@"
