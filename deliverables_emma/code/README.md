# EE6483 Project 2 — Dogs vs Cats & CIFAR-10

Two-part coursework project:

* **Part 1** — binary classification of dog/cat images, `submission.csv` on 500 test images.
* **Part 2** — transfer the pipeline to 10-way CIFAR-10, then study **class imbalance**.

## Environment

```bash
pip install -r requirements.txt
```

CUDA is used when available (`torch.cuda.is_available()`); everything also runs on CPU
(slower). All runs fix the random seed; `--deterministic` additionally disables
cuDNN autotuning at a small speed cost.

## Data layout

```
dataset/
├── train/{cat,dog}/     10,000 images each
├── val/{cat,dog}/        2,500 images each
└── test/                  500 flat files: 1.jpg ... 500.jpg
```

`ImageFolder` sorts class names alphabetically, so `cat = 0`, `dog = 1`, which is exactly
the label convention required by `sampleSubmission.csv` (`1 = dog`, `0 = cat`).

CIFAR-10 is read with `torchvision.datasets.CIFAR10(data_root, download=False)`, where
`data_root` must contain the `cifar-10-batches-py/` folder.

## Quick start

```bash
bash scripts/run_smoke.sh       # ~2 min, verifies the whole pipeline on a small subset
bash scripts/run_baseline.sh    # main model -> submission.csv + error analysis + CIFAR-10
bash scripts/run_experiments.sh # ablation table for report items c) and f)
bash scripts/run_cifar.sh       # CIFAR-10 + four imbalance strategies
```

## Files

| File | Purpose | Report item |
| --- | --- | --- |
| `data.py` | transforms, `ImageFolder` loaders, flat test-set dataset | b |
| `model.py` | ResNet-18/50, ConvNeXt-Tiny, CIFAR stem ResNet + linear head | b |
| `trainer.py` | generic train/eval loops (AMP, frozen backbone), metrics | b |
| `train.py` | Dogs vs Cats training, saves best checkpoint + history | b, c, d |
| `predict.py` | writes `submission.csv` (optional h-flip TTA) + sanity checks | d |
| `analyze.py` | misclassified/correct sample grids, confidence histogram | e |
| `cifar10.py` | CIFAR-10 with a held-out validation split | g |
| `imbalance.py` | long-tailed CIFAR-10; baseline / sampler / weighted CE / focal | h |
| `figures.py` | training curves, comparison bar chart, confusion matrices, `summary.csv` | c, e, f, g, h |

## Design notes

* **Validation discipline.** The CIFAR-10 test split is never used for model selection:
  5,000 of the 50,000 training images are held out as validation, and the test set is
  evaluated exactly once. For Dogs vs Cats the provided `val/` split is used, and the
  unlabelled `test/` split is only used to produce `submission.csv`.
* **Error analysis uses validation.** The test images have no labels, so qualitative
  error analysis is performed on the validation split (stated in the report).
* **Two-stage training.** The ImageNet backbone is frozen for the first epoch(s) so the
  randomly initialised head can settle, then the whole network is fine-tuned with a
  smaller learning rate for the backbone.
* **AMP** halves memory and roughly speeds training up; the model stays in fp32 weights.

## Outputs

* `runs/<run>/best.pt`, `history.json`, `result.json` — per-run artefacts
* `submission.csv` — 500 rows, `id,label`
* `analysis/` — `wrong_grid.png`, `right_grid.png`, `confidence_hist.png`, `wrong_samples.csv`
* `report_figures/` — `curves_*.png`, `compare.png`, `cm_*.png`, `summary.csv`
