#!/bin/bash
cd "$(dirname "$0")/.."
python3 cifar10.py --data_root ~/ee6483_data/cifar10 --arch resnet18_cifar \
  --img_size 32 --epochs 20 --batch_size 256 --num_workers 6 --out_dir runs/cifar10
