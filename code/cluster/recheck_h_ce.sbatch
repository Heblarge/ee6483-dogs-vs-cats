#!/bin/bash
# Reproduce the fixed H_ce baseline, then evaluate post-hoc logit adjustment.
# This is a verification run, not a hyper-parameter search: the 18-epoch recipe,
# seed 42 and tau candidate set are frozen in run_experiments.py/eval_logit_adjust.py.
#
# Submit from code/ with:
#   sbatch --export=ALL,P=/projects/embodied/zihanshu001 cluster/recheck_h_ce.sbatch
#
# Use an isolated checkout when preserving earlier result files matters.
#SBATCH --job-name=ee6483-hce-check
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh

module load Miniforge3
set +u
source activate "$P/envs/ee6483"
source "$P/envs/ee6483-recheck/bin/activate"
set -u

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available()); assert torch.cuda.is_available()'

python -m ee6483.src.train_cifar \
  --mode h --method ce --epochs 18 --exp-id H_ce \
  --seed 42 --workers 4 --data-root data/cifar

python -m ee6483.scripts.eval_logit_adjust \
  --ce-ckpt ee6483/results/H_ce/s42/best.pt \
  --data-root data/cifar \
  --out ee6483/results/H_logit_adjust/s42/metrics.json

python -m ee6483.scripts.aggregate
