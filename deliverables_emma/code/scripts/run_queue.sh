#!/bin/bash
# Overnight experiment queue for EE6483 Project 2.
# - waits for CIFAR-10 to finish
# - runs 4 imbalance strategies, then the Dogs-vs-Cats ablations
# - skips any run that already has result.json (safe to re-run)
# - never aborts the queue on a single failure
cd "$(dirname "$0")/.." || exit 1
LOG_DIR=../logs
mkdir -p "$LOG_DIR" runs

say() { echo "[queue $(date +%H:%M:%S)] $*"; }

run_if_missing() {
  local out="$1"; shift
  if [ -f "runs/$out/result.json" ]; then
    say "skip $out (already done)"
    return 0
  fi
  say "START $out"
  if "$@" > "$LOG_DIR/$out.log" 2>&1; then
    say "DONE  $out  $(python3 -c "import json;d=json.load(open('runs/$out/result.json'));print({k:v for k,v in d.items() if k in ('best_val_acc','test_acc','macro_f1','minority_avg_acc')})" 2>/dev/null)"
  else
    say "FAIL  $out (see $LOG_DIR/$out.log) — continuing"
  fi
}

# 1) wait for the running CIFAR-10 job ([c] trick avoids matching this script)
while pgrep -f "[c]ifar10.py --data_root" > /dev/null; do sleep 30; done
say "CIFAR-10 finished, starting the queue"

# 1.5) CIFAR-10 with an ImageNet-pretrained backbone (higher accuracy reference)
run_if_missing cifar10_pretrained python3 cifar10.py \
  --data_root "$HOME/ee6483_data/cifar10" --arch resnet18 --img_size 64 \
  --epochs 15 --batch_size 128 --num_workers 6 --out_dir runs/cifar10_pretrained

# 2) class-imbalance experiments (report item h)
for m in baseline sampler weighted_ce focal; do
  run_if_missing "imbalance_$m" python3 imbalance.py \
    --data_root "$HOME/ee6483_data/cifar10" --method "$m" \
    --epochs 10 --batch_size 256 --num_workers 6 --out_dir "runs/imbalance_$m"
done

# 3) Dogs vs Cats ablations (report items c and f)
run_if_missing e2_resnet50 python3 train.py --data_root "$HOME/ee6483_data/dataset" \
  --arch resnet50 --img_size 160 --epochs 8 --batch_size 16 --out_dir runs/e2_resnet50

run_if_missing e3_convnext python3 train.py --data_root "$HOME/ee6483_data/dataset" \
  --arch convnext_tiny --img_size 160 --epochs 8 --batch_size 16 --lr 5e-4 --out_dir runs/e3_convnext

run_if_missing e7_scratch python3 train.py --data_root "$HOME/ee6483_data/dataset" \
  --arch resnet18 --img_size 160 --epochs 8 --no_pretrained --freeze_epochs 0 --out_dir runs/e7_scratch

run_if_missing e5_strongaug python3 train.py --data_root "$HOME/ee6483_data/dataset" \
  --arch resnet18 --img_size 160 --epochs 8 --strong --out_dir runs/e5_strongaug

run_if_missing e6_img112 python3 train.py --data_root "$HOME/ee6483_data/dataset" \
  --arch resnet18 --img_size 112 --epochs 8 --out_dir runs/e6_img112

run_if_missing e4_lr5e4 python3 train.py --data_root "$HOME/ee6483_data/dataset" \
  --arch resnet18 --img_size 160 --epochs 8 --lr 5e-4 --out_dir runs/e4_lr5e4

# 4) extra seeds for mean +- std (reproducibility requirement)
for s in 43 44; do
  run_if_missing "seed$s" python3 train.py --data_root "$HOME/ee6483_data/dataset" \
    --arch resnet18 --img_size 160 --epochs 15 --seed "$s" --out_dir "runs/seed$s"
done

# 5) refresh all figures / summary tables
python3 figures.py --runs runs --out report_figures > "$LOG_DIR/figures.log" 2>&1
say "queue complete"
