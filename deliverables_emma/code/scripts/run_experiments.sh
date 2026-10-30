#!/bin/bash
# Report items c) and f): model / hyper-parameter / augmentation comparisons.
set -e
python train.py --arch resnet18  --img_size 160 --epochs 15 --out_dir runs/e1_resnet18
python train.py --arch resnet50  --img_size 160 --epochs 15 --batch_size 16 --out_dir runs/e2_resnet50
python train.py --arch convnext_tiny --img_size 160 --epochs 12 --batch_size 16 \
  --lr 5e-4 --out_dir runs/e3_convnext
python train.py --arch resnet18 --img_size 160 --epochs 15 --lr 5e-4 --out_dir runs/e4_lr5e4
python train.py --arch resnet18 --img_size 160 --epochs 15 --strong --out_dir runs/e5_strongaug
python train.py --arch resnet18 --img_size 112 --epochs 15 --out_dir runs/e6_img112
python train.py --arch resnet18 --img_size 160 --epochs 15 --no_pretrained \
  --freeze_epochs 0 --out_dir runs/e7_scratch
for s in 43 44; do
  python train.py --arch resnet18 --img_size 160 --epochs 15 --seed $s --out_dir runs/seed$s
done
python figures.py --runs runs --out report_figures
