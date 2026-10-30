#!/bin/bash
# wait for the e0 run to finish, then produce submission + analysis + figures
cd "$(dirname "$0")/.."
while pgrep -f "train.py.*e0_resnet18_160" > /dev/null; do sleep 20; done
echo "[chain] training finished at $(date +%H:%M:%S)"
python3 predict.py --ckpt runs/e0_resnet18_160/best.pt --out submission.csv --tta
python3 analyze.py --ckpt runs/e0_resnet18_160/best.pt --out_dir analysis
python3 figures.py --runs runs --out report_figures
echo "[chain] done $(date +%H:%M:%S)"
