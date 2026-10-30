#!/bin/bash
# Report items g) and h)
set -e
python cifar10.py --arch resnet18_cifar --img_size 32 --epochs 30 --out_dir runs/cifar10
for m in baseline sampler weighted_ce focal; do
  python imbalance.py --method $m --out_dir runs/imbalance
done
python figures.py --runs runs --out report_figures
