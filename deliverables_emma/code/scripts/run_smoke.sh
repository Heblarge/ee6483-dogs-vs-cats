#!/bin/bash
# 2-minute sanity check of the whole pipeline on a tiny subset
set -e
python train.py --arch resnet18 --img_size 128 --epochs 2 --batch_size 32 \
  --max_train_per_class 60 --max_val_per_class 30 --out_dir runs/smoke
python predict.py --ckpt runs/smoke/best.pt --out runs/smoke/submission.csv
python analyze.py --ckpt runs/smoke/best.pt --out_dir runs/smoke/analysis
