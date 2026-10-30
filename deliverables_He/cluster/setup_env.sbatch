#!/bin/bash
# 一次性：建 conda 环境 + 下载 7 个模型的预训练权重。CPU 任务（免费），登录节点上不允许装包。
# 在 deliverables_He/ 下提交：sbatch cluster/setup_env.sbatch
#SBATCH --job-name=ee6483-setup
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=cluster/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh

module load Miniforge3
[ -d "$ENV" ] || conda create -y -p "$ENV" python=3.12
source activate "$ENV"
# cu128 同时支持 a40 / l40 和 Blackwell 的 rtx5090 / pro6000（后两者要求 CUDA ≥ 12.8）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install numpy pillow pyyaml tqdm

python - <<'PY'
import sys; sys.path.insert(0, ".")
from src.policies import build_model
for n in ["resnet18", "resnet34", "resnet50", "efficientnet_b0", "convnext_tiny", "swin_t", "vit_b_16"]:
    build_model({"name": n, "pretrained": True}, 2)
    print("权重已缓存:", n)
PY
python -c "import torch, torchvision; print('torch', torch.__version__, 'torchvision', torchvision.__version__)"
