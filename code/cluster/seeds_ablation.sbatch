#!/bin/bash
# REMAINING_WORK B2 (targeted) + B3:
#   B2  seeds 43/44 for the four fine-tuning strategies of Section 4.4 (E7_ft_head,
#       E7_ft_last, E7_ft_lpft, E7_aug_full = full fine-tuning). Arguments identical to
#       run_experiments.py PHASES["ablation"]; seed 42 results are not touched.
#   B3  clean-validation accuracy of the submitted ensemble (inference only).
# Expects data/cache (built here if missing) and the six backbone checkpoints under runs/.
#   sbatch --export=ALL,P=/projects/embodied/zihanshu001 cluster/seeds_ablation.sbatch
#SBATCH --job-name=ee6483-b2b3
#SBATCH --gres=gpu:a40:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
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

python scripts/clean_val_ensemble.py --data-root ../datasets/datasets --out results/clean_val.json

[ -d data/cache/val ] || python -m ee6483.src.data.cache_images --src ../datasets/datasets --dst data/cache --short 256
COMMON="--model resnet18 --epochs 3 --lr 3e-4 --head-lr 1e-3 --batch 64 --img-size 160 --aug full
  --n-per-class 0 --pretrained 1 --data-root data/cache --workers 4 --patience 3"
for s in 43 44; do
  python -m ee6483.src.train --exp-id E7_ft_head  --ft-strategy head       --freeze-epochs 2 --seed $s $COMMON
  python -m ee6483.src.train --exp-id E7_ft_last  --ft-strategy last_stage --freeze-epochs 2 --seed $s $COMMON
  python -m ee6483.src.train --exp-id E7_ft_lpft  --ft-strategy lpft       --freeze-epochs 1 --seed $s $COMMON
  python -m ee6483.src.train --exp-id E7_aug_full --ft-strategy full       --freeze-epochs 2 --seed $s $COMMON
done
echo B2B3 DONE
