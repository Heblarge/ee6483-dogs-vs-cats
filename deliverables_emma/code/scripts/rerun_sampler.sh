#!/bin/bash
cd "$(dirname "$0")/.."
# 等 seed44 跑完再开，避免 2GB 显存争抢
while pgrep -f "[t]rain.py.*seed44" > /dev/null; do sleep 30; done
python3 imbalance.py --data_root "$HOME/ee6483_data/cifar10" --method sampler \
  --epochs 10 --batch_size 256 --num_workers 6 --out_dir runs/imbalance_sampler
