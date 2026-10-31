# Dogs vs. Cats — EE6483 Mini Project (Option 2)

> Guide to the report and the code that produced it (`ee6483/`, `report/`, `submissions/`).
> All commands below are run from inside `deliverables_He/`.
>
> `deliverables_He/` also contains a second, config-driven framework (`src/`, `scripts/`,
> `configs/`, `cluster/`) used for the backbone comparison in `report/backbone_comparison.md`;
> see `README.md`. The two code bases are independent.

Binary image classification of cats and dogs, with a full experimental study of transfer
learning, an ablation matrix, a CIFAR-10 extension and a class-imbalance comparison.

**Result:** 99.70% validation accuracy for the submitted ensemble (ConvNeXt-T, ResNet-50 and Swin-T,
two seeds each, horizontal-flip TTA; report Sections 5.3 and 7.8). The submitted
`submissions/submission.csv` contains 500 predictions (259 dog / 241 cat). The first submission
(ResNet-18 ×3 seeds + TTA, 99.22%) is kept as `submissions/submission_m3_ensemble.csv`.

| Model | Validation accuracy |
|---|---|
| HOG + SVM (classical baseline) | 75.00% |
| SmallCNN from scratch | 95.06% |
| ResNet-18, frozen backbone + linear head | 97.78% |
| **ResNet-18, LP-FT (main model)** | **98.93 ± 0.05%** (3 seeds) |
| ResNet-18 ×3 seeds + TTA (first submission) | 99.22% |
| **ConvNeXt-T + ResNet-50 + Swin-T, ×2 seeds + TTA (submitted)** | **99.70%** |

---

## Repository layout

```
.
├── REPORT_GUIDE.md            this file
├── requirements.txt           Python dependencies (both code bases)
├── submissions/
│   ├── submission.csv         the 500 test-set predictions (id, label; 1=dog, 0=cat)
│   ├── sampleSubmission.csv   the format template provided with the assignment
│   └── submission.probs.npy   per-class probabilities behind submission.csv
│
├── docs/                      assignment documents and planning notes
├── report/                    the report, its sources and the build scripts
└── ee6483/                    all code, results and figures
```

`data/` is **not** in the repository (3 GB). See [Data](#data) below.

---

## What every file does

### `ee6483/` — code package

| File | Purpose |
|---|---|
| `README.md` | Code-level reproduction guide (data layout, class mapping, step-by-step commands) |
| `configs/main.yaml` | Example configuration for the main model; CLI arguments override it |

### `ee6483/src/` — the library

| File | Purpose |
|---|---|
| `utils.py` | Seeding (`set_seed`), logging, metric IO, yaml config merge, parameter counting |
| `engine.py` | `train_one_epoch` / `evaluate`, shared by every experiment |
| `losses.py` | Cross-entropy with label smoothing, class-balanced loss, LDAM, logit adjustment |
| `train.py` | **Single-run trainer for the dog/cat task** (CLI; writes `metrics.json`, `curves.csv`, `train.log`, `best.pt`) |
| `train_cifar.py` | CIFAR-10 trainer for both the multi-class task (G) and the imbalance study (H) |
| `predict.py` | Builds `submission.csv` from one or more checkpoints (ensemble + hflip TTA) |
| `classical.py` | M0 baseline: HOG + colour histogram → SVM (RBF and linear) |
| `gradcam.py` | Grad-CAM visualisation for the error analysis |
| `analyze.py` | All dog/cat figures: dataset samples, augmentation, curves, confusion, errors, robustness, test confidence |
| `analyze_cifar.py` | All CIFAR-10 and imbalance figures |
| `figstyle.py` | Shared matplotlib style (serif fonts, 300 dpi) so figures match the report |
| `models/backbones.py` | Model zoo: torchvision ResNets, CIFAR stem variant, freeze/unfreeze helpers |
| `models/small_cnn.py` | The 1.17M-parameter from-scratch CNN (M1) |
| `data/transforms.py` | Pre-processing and the five augmentation levels (`none`/`flip`/`geo`/`color`/`full`) |
| `data/dvc_dataset.py` | Dataset construction, stratified splits, nested class-balanced subsets |
| `data/cache_images.py` | One-off resize cache (short side 256 px) so every run sees identical pixels |
| `data/check_data.py` | Dataset health check: counts, sizes, corrupt files, id mapping |
| `data/cifar_parquet.py` | CIFAR-10 loader backed by the HuggingFace parquet mirror (the official server is very slow) |

### `ee6483/scripts/` — experiment orchestration and audits

| File | Purpose |
|---|---|
| `run_experiments.py` | Runs the whole matrix phase by phase (`main`, `compare`, `ablation`, `lrsearch`, `cifar`, `imbalance`); skips runs that already have a `metrics.json` |
| `aggregate.py` | Collects every `results/**/metrics.json` into `summary.csv` and `summary.md` |
| `validate_submission.py` | Strict format check for `submission.csv` — run before every upload |
| `eval_ensemble.py` | Evaluates the seed ensemble (+TTA) on the validation set |
| `eval_logit_adjust.py` | Post-hoc logit adjustment on the CE checkpoint, no retraining |
| `find_duplicates.py` | pHash near-duplicate audit at a Hamming radius (found 40 pairs and 6 placeholder images) |
| `dump_errors.py` | Exports every validation error as an image grid |
| `make_arch_fig.py` | Draws the architecture figure required by the report |
| `make_dup_figure.py` | Draws the near-duplicate / placeholder figure |

### `ee6483/results/` — the experiment log (the evidence for every number)

| Path | Purpose |
|---|---|
| `<exp>/s<seed>/metrics.json` | Every scalar result for that run (accuracy, F1, confusion matrix, per-class recall, args, timings) |
| `<exp>/s<seed>/curves.csv` | Per-epoch loss and accuracy |
| `<exp>/s<seed>/train.log` | Full training log |
| `summary.csv` / `summary.md` | All runs in one table |
| `run_log.txt` | Every command line that was executed |
| `data_report.json` | Output of the dataset health check |
| `duplicates_all.json` | Every near-duplicate pair found |
| `phash_cache.npz` | Cached perceptual hashes (regenerable, git-ignored) |

Checkpoints (`best.pt`, 1.8 GB total) are git-ignored; each is reproduced by the
command recorded in `run_log.txt`.

### `ee6483/figures/` — 24 figures

18 are used in the report; the remaining six are supplementary (`fig_errors_all_1..5.png`
show all 50 validation errors, `fig_error_cases.png` is an earlier layout).

### `report/` — the report and its build pipeline

| File | Purpose |
|---|---|
| `EE6483_Group_Report.docx` | **The submitted report** (A4, 10 sections + references + appendix) |
| `REPORT.md` | The same report as Markdown (generated) |
| `sections/*.md` | The source of each section, one file per section |
| `make_tokens.py` | Reads `results/**/metrics.json` and writes `tokens.json` — every number in the report comes from here |
| `build_report.py` | Assembles `sections/*.md` into `REPORT.md`, substitutes the tokens, numbers the figures |
| `md_to_docx.py` | Converts `REPORT.md` into the `.docx` (headings, tables, figures, LaTeX→Unicode) |
| `tokens.json` | The resolved values used in the last build |

The report is generated, not hand-assembled: re-running the pipeline after new experiments
updates every table and every number quoted in the prose.

### `docs/` — assignment material

| File | Purpose |
|---|---|
| `EE6483-Project2.pdf` | The assignment brief |
| `EE6483_project_briefing.pdf` | The briefing slides |
| `BEST_PLAN.md` | The execution plan, with a "plan vs actual" comparison at the top |
| `PROJECT_PLAN.md` | The first version of the plan (superseded, kept for history) |

---

## Data

The dataset is not included. It is distributed through the link in `docs/EE6483-Project2.pdf`:

```
data/datasets/train/{cat,dog}   10,000 images each
data/datasets/val/{cat,dog}      2,500 images each
data/datasets/test/                500 images, named 1.jpg … 500.jpg
```

After downloading, build the resize cache once (it makes every experiment reproducible):

```bash
python -m ee6483.src.data.cache_images --src data/datasets --dst data/cache --short 256
```

The paths above are relative to this folder. In this repository the dataset sits at the
repository root (`../datasets/datasets`, shared with the other code base), so pass that path
instead: `--src ../datasets/datasets --dst data/cache`.

CIFAR-10 is downloaded automatically from the HuggingFace parquet mirror on first use.

## Quick start

```bash
pip install -r requirements.txt          # see the note about the CUDA build inside

# 1. reproduce the submitted predictions (three seeds, then the ensemble)
python -m ee6483.src.train --config ee6483/configs/main.yaml --exp-id E4_r18_ft --seed 42
python -m ee6483.src.train --config ee6483/configs/main.yaml --exp-id E4_r18_ft --seed 43
python -m ee6483.src.train --config ee6483/configs/main.yaml --exp-id E4_r18_ft --seed 44
python -m ee6483.src.predict --ckpts ee6483/results/E4_r18_ft/s4{2,3,4}/best.pt \
    --out submissions/submission.csv --tta 1
python ee6483/scripts/validate_submission.py --sub submissions/submission.csv

# 2. regenerate every table and figure
python -m ee6483.scripts.run_experiments --phases main compare ablation lrsearch cifar imbalance
python -m ee6483.scripts.aggregate
python -m ee6483.src.analyze --which all
python -m ee6483.src.analyze_cifar

# 3. regenerate the report
python report/make_tokens.py
python report/build_report.py
python report/md_to_docx.py
```

## Environment

Python 3.13, PyTorch 2.6.0+cu124, torchvision 0.21.0, one RTX 3070 Ti Laptop GPU (8 GB).
The full experiment matrix (42 runs) takes about 3.7 hours. Exact versions are listed in
Appendix A.2 of the report.

## Licence and attribution

Course project for EE6483 (NTU). The dataset belongs to its original providers and the
pre-trained weights come from torchvision. The code in this repository was written for
this project.
