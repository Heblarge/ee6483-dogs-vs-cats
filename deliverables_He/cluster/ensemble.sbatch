#!/bin/bash
# 骨干对比的 checkpoint 集成预测：验证集上比较候选方案，再用第一梯队 6 个模型 + 水平翻转 TTA 生成 submission。
# 在 deliverables_He/ 下提交（需先跑完 backbones.sbatch）：sbatch cluster/ensemble.sbatch
# 输出：submissions/submission_backbone_ensemble{.csv,_probs.csv,_ensemble.json}
#SBATCH --job-name=ee6483-ens
#SBATCH --gres=gpu:a40:1
#SBATCH --time=0:40:00
#SBATCH --output=cluster/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh

module load Miniforge3
set +u; source activate "$ENV"; set -u   # conda 的激活脚本不兼容 set -u

c() { echo "runs/backbones/b_$1_s0/checkpoints/best.pt,runs/backbones/b_$1_s1/checkpoints/best.pt"; }
# 候选方案事先定好；最终采用 top3（第一梯队 3 种网络 × 2 seeds），其余两组只作对照
python scripts/ensemble_predict.py \
  --group convnext="$(c convnext_tiny)" \
  --group resnet18="$(c resnet18)" \
  --group top3="$(c convnext_tiny),$(c resnet50),$(c swin_t)" \
  --choose top3 --tta --out submissions/submission_backbone_ensemble.csv
