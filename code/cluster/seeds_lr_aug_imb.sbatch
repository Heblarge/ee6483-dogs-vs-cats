#!/bin/bash
# Seeds 43/44 for the learning-rate sweep and the augmentation budget check of Section 7
# (PHASES["lrsearch"], PHASES["augcheck"]) and for the decoupling and class-balanced rows of
# Section 9.4 (H_decouple, H_cb, H_cb_long). Recipes are read from run_experiments.py; only
# --seed changes. Needs data/cache and data/cifar.
#   sbatch --export=ALL,P=/projects/embodied/zihanshu001 cluster/seeds_lr_aug_imb.sbatch
#SBATCH --job-name=ee6483-lr-imb-seeds
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:a40:1
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
python -m ee6483.scripts.run_extra_seeds --phases imbalance --only H_decouple H_cb H_cb_long --seeds 43 44
python -m ee6483.scripts.run_extra_seeds --phases lrsearch augcheck --seeds 43 44
echo LRIMB DONE
