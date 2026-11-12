#!/bin/bash
# REMAINING_WORK B1: extra seeds (43, 44) for the CE baseline, Re-sampling and post-hoc
# logit adjustment on CIFAR-10-LT. Recipe identical to run_experiments.py PHASES["imbalance"];
# seed 42 results are not touched (new runs land in H_*/s43, H_*/s44).
#   sbatch --export=ALL,P=/projects/embodied/zihanshu001 cluster/seeds_imbalance.sbatch
#SBATCH --job-name=ee6483-b1-seeds
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:30:00
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
nvidia-smi --query-gpu=name --format=csv,noheader
for s in 43 44; do
  for m in ce resample; do
    python -m ee6483.src.train_cifar --mode h --method $m --epochs 18 --exp-id H_$m \
      --seed $s --workers 4 --data-root data/cifar
  done
  python -m ee6483.scripts.eval_logit_adjust --ce-ckpt ee6483/results/H_ce/s$s/best.pt \
    --data-root data/cifar --out ee6483/results/H_logit_adjust/s$s/metrics.json
done
echo B1 DONE
