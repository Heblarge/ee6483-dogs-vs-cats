#!/bin/bash
# E0: reference baseline -> submission.csv + error analysis + figures
set -e
python train.py --arch resnet18 --img_size 160 --epochs 15 --batch_size 32 \
  --lr 1e-3 --out_dir runs/e0_resnet18_160
python predict.py --ckpt runs/e0_resnet18_160/best.pt --out submission.csv --tta
python analyze.py --ckpt runs/e0_resnet18_160/best.pt --out_dir analysis
python cifar10.py --arch resnet18_cifar --img_size 32 --epochs 30 --out_dir runs/cifar10
python figures.py --runs runs --out report_figures
