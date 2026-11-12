#!/bin/bash
# Seeds 43/44 for the remaining Section 7 one-factor ablation runs (PHASES["ablation"] minus the
# four fine-tuning strategies already covered by seeds_ablation.sbatch). Recipes are read from
# run_experiments.py; only --seed changes. Needs data/cache.
#   sbatch --export=ALL,P=/projects/embodied/zihanshu001 cluster/seeds_ablation_rest.sbatch
#SBATCH --job-name=ee6483-abl-seeds
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
python -m ee6483.scripts.run_extra_seeds --phases ablation --seeds 43 44 \
  --skip E7_ft_head E7_ft_last E7_ft_lpft E7_aug_full
echo ABL DONE
