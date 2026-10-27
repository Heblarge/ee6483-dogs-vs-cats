# EE6483 Mini Project — Dogs vs Cats (Option 2)

Code package for the Dogs vs Cats binary classification project. The repository-level
overview, the report and the submission file are described in `../README.md`.
Everything is driven by command line arguments; every run writes the same
`metrics.json` schema so results can be aggregated automatically.

## 1. Environment

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r ../requirements.txt                # requirements.txt lives at the repo root
```

Tested with Python 3.13, PyTorch 2.6.0+cu124, torchvision 0.21.0 on a single
RTX 3070 Ti Laptop (8 GB). CPU-only training works but is ~40x slower.

## 2. Data layout

```
data/
├─ datasets/            # as provided on the course Google Drive
│  ├─ train/{cat,dog}   # 10,000 images each
│  ├─ val/{cat,dog}     #  2,500 images each
│  └─ test/*.jpg        #    500 images, file name == submission id
└─ cache/               # produced by step 3 (short side 256 px)
```

`ImageFolder` sorts class names alphabetically, so `cat=0, dog=1`, which is
exactly the label convention of `sampleSubmission.csv`. The loader asserts this
at runtime (`check_class_mapping`).

## 3. Reproduce

```bash
# 0) dataset health check (counts, corrupt files, id mapping, near-duplicates)
python -m ee6483.src.data.check_data --root data/datasets

# 1) offline resize cache (cuts epoch time ~2x)
python -m ee6483.src.data.cache_images --src data/datasets --dst data/cache --short 256

# 2) main model, 3 seeds (the settings are in configs/main.yaml; CLI overrides yaml)
for s in 42 43 44; do
  python -m ee6483.src.train --config ee6483/configs/main.yaml \
      --exp-id E4_r18_ft --seed $s --data-root data/cache
done

# 3) submission (3-seed ensemble + horizontal-flip TTA)
python -m ee6483.src.predict \
    --ckpts ee6483/results/E4_r18_ft/s42/best.pt \
            ee6483/results/E4_r18_ft/s43/best.pt \
            ee6483/results/E4_r18_ft/s44/best.pt \
    --out submission.csv --tta 1

# 4) validate before uploading
python ee6483/scripts/validate_submission.py --sub submission.csv

# 5) everything else: comparisons, ablations, learning-rate sweep, CIFAR-10, imbalance, figures
python -m ee6483.scripts.run_experiments \
    --phases main compare ablation lrsearch cifar imbalance extra figure
python -m ee6483.scripts.aggregate            # -> results/summary.{csv,md}
python -m ee6483.src.analyze_cifar            # CIFAR / imbalance figures
python -m ee6483.scripts.find_duplicates      # near-duplicate audit (needs imagehash)
```

## 4. Layout

```
ee6483/
├─ src/
│  ├─ data/        transforms, ImageFolder loaders, cache, health check
│  ├─ models/      backbones (torchvision), SmallCNN, CIFAR ResNet
│  ├─ losses.py    CE / label smoothing / class-balanced / LDAM / logit adjustment
│  ├─ train.py     single-run trainer (argparse + optional yaml)
│  ├─ train_cifar.py  Part G (CIFAR-10) and Part H (imbalance)
│  ├─ engine.py    train/eval loops, metric computation
│  ├─ predict.py   ensemble + TTA -> submission.csv
│  ├─ gradcam.py   Grad-CAM for error analysis
│  └─ analyze*.py  all report figures
├─ scripts/        orchestrator, aggregation, submission validator
├─ results/<exp>/s<seed>/{metrics.json, curves.csv, train.log, best.pt}
└─ figures/
```

## 5. Reproducibility

* `set_seed(42/43/44)` fixes python, numpy, torch and cuda RNGs;
  `cudnn.deterministic=True`, `benchmark=False`.
* AMP kernels are not bit-exact, so every headline number is reported as
  mean ± std over three seeds rather than a single run.
* Each run stores its full argument dictionary inside `metrics.json` and
  `best.pt`, and the orchestrator appends the exact command line to
  `results/run_log.txt`.

## 6. Measured results (this hardware)

| experiment | val acc | time |
|---|---|---|
| HOG + SVM (8k images) | 75.00% | ~8 min |
| SmallCNN from scratch @128, 14 ep | 95.06% | ~8 min |
| ResNet18 frozen + linear head, 6 ep | 97.78% | ~6 min |
| **ResNet18 fine-tuned @224 (LP-FT), 8 ep** | **98.93% ± 0.05** (3 seeds) | ~8 min/seed |
| ResNet18 ×3 seeds + hflip TTA | **99.22%** | inference only |
| ResNet34 fine-tuned @224 | 97.54% | ~11 min |
| ResNet18 from scratch @160, 6 ep (ablation) | 90.66% | ~4 min |
| CIFAR-10 ResNet18-CIFAR from scratch, 25 ep | see `results/G1_cifar_scratch` | ~25 min |
| CIFAR-10 ResNet18 ImageNet @160, 7 ep | see `results/G2_cifar_pretrained` | ~13 min |

Full tables and the discussion are in `report/REPORT.md`.

