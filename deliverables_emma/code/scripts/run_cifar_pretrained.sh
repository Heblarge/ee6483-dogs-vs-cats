#!/bin/bash
cd "$(dirname "$0")/.."
python3 cifar10.py --data_root ~/ee6483_data/cifar10 --arch resnet18 \
  --img_size 64 --epochs 15 --batch_size 128 --num_workers 6 \
  --out_dir runs/cifar10_pretrained
