# Appendix: Complete Experiment Log

Everything reported in the body of this document is derived from the 42 run
directories under `ee6483/results/` and the backbone runs of A.1b. This appendix
prints the per-run results so that any number in the main text can be traced to
a run directory; the commands are in `ee6483/scripts/run_experiments.py` and,
for the runs launched by hand, in the notes below.

## A.1 Every run

Columns: `exp` is the experiment id (the folder name), `seed` the random seed,
`n_train` the number of training images actually used, `epochs` the number of
epochs completed (early stopping may cut the schedule short), `val_acc` the best
validation accuracy, `test_acc` the CIFAR-10 test accuracy where applicable, and
`sec` the wall-clock training time in seconds on the RTX 3070 Ti Laptop.

| exp | seed | model | img | aug | ft | n_train | ep | val acc (%) | test acc (%) | time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| `E2_smallcnn` | 42 | smallcnn | 128 | full | scratch | 20,000 | 14 | 95.06 | – | 454 |
| `E3_r18_frozen` | 42 | resnet18 | 224 | full | head | 20,000 | 6 | 97.78 | – | 334 |
| `E4_r18_ft` | 42 | resnet18 | 224 | full | lpft | 20,000 | 8 | 99.00 | – | 522 |
| `E4_r18_ft` | 43 | resnet18 | 224 | full | lpft | 20,000 | 8 | 98.90 | – | 498 |
| `E4_r18_ft` | 44 | resnet18 | 224 | full | lpft | 20,000 | 8 | 98.90 | – | 459 |
| `E5_r34_ft` | 42 | resnet34 | 224 | full | lpft | 20,000 | 4 | 97.54 | – | 287 |
| `E6_r50_ft` | 42 | resnet50 | 160 | full | lpft | 20,000 | 6 | 99.30 | – | 318 |
| `E7_aug_color` | 42 | resnet18 | 160 | color | full | 20,000 | 3 | 98.48 | – | 149 |
| `E7_aug_flip` | 42 | resnet18 | 160 | flip | full | 20,000 | 3 | 98.62 | – | 134 |
| `E7_aug_full` | 42 | resnet18 | 160 | full | full | 20,000 | 3 | 98.12 | – | 177 |
| `E7_aug_geo` | 42 | resnet18 | 160 | geo | full | 20,000 | 3 | 98.56 | – | 120 |
| `E7_aug_none` | 42 | resnet18 | 160 | none | full | 20,000 | 3 | 98.36 | – | 132 |
| `E7_data_1000` | 42 | resnet18 | 160 | full | full | 2000 | 3 | 96.76 | – | 96 |
| `E7_data_250` | 42 | resnet18 | 160 | full | full | 500 | 3 | 93.56 | – | 89 |
| `E7_data_2500` | 42 | resnet18 | 160 | full | full | 5000 | 3 | 97.90 | – | 105 |
| `E7_data_5000` | 42 | resnet18 | 160 | full | full | 10000 | 3 | 98.20 | – | 117 |
| `E7_ft_head` | 42 | resnet18 | 160 | full | head | 20,000 | 3 | 96.24 | – | 154 |
| `E7_ft_last` | 42 | resnet18 | 160 | full | last_stage | 20,000 | 3 | 98.20 | – | 165 |
| `E7_ft_lpft` | 42 | resnet18 | 160 | full | lpft | 20,000 | 3 | 98.32 | – | 174 |
| `E7_lr_1e-3` | 42 | resnet18 | 160 | full | full | 20,000 | 3 | 97.70 | – | 207 |
| `E7_lr_1e-4` | 42 | resnet18 | 160 | full | full | 20,000 | 3 | 98.62 | – | 210 |
| `E7_lr_1e-5` | 42 | resnet18 | 160 | full | full | 20,000 | 3 | 97.68 | – | 215 |
| `E7_lr_3e-3` | 42 | resnet18 | 160 | full | full | 20,000 | 3 | 96.06 | – | 207 |
| `E7_lr_3e-4` | 42 | resnet18 | 160 | full | full | 20,000 | 3 | 98.24 | – | 212 |
| `E7_res_128` | 42 | resnet18 | 128 | full | full | 20,000 | 3 | 97.94 | – | 167 |
| `E7_res_224` | 42 | resnet18 | 224 | full | full | 20,000 | 3 | 98.76 | – | 235 |
| `E7_res_96` | 42 | resnet18 | 96 | full | full | 20,000 | 3 | 97.08 | – | 148 |
| `E7_scratch` | 42 | resnet18 | 160 | full | scratch | 20,000 | 6 | 90.66 | – | 341 |
| `E7b_aug_full_8ep` | 42 | resnet18 | 160 | full | full | 20,000 | 8 | 98.66 | – | 323 |
| `E7b_aug_none_8ep` | 42 | resnet18 | 160 | none | full | 20,000 | 8 | 98.42 | – | 241 |
| `E7c_aug_full_lowdata` | 42 | resnet18 | 160 | full | full | 500 | 8 | 96.30 | – | 88 |
| `E7c_aug_none_lowdata` | 42 | resnet18 | 160 | none | full | 500 | 8 | 95.94 | – | 87 |
| `G1_cifar_scratch` | 42 | resnet18-cifar | 32 | - | ce | 45,000 | 25 | 93.44 | 93.60 | 1540 |
| `G2_cifar_pretrained` | 42 | resnet18 (ImageNet) | 160 | - | ce | 45,000 | 7 | 95.58 | 95.20 | 579 |
| `H_cb` | 42 | resnet18-cifar | 32 | - | cb | 11,908 | 18 | 54.02 | 54.04 | 658 |
| `H_cb_long` | 42 | resnet18-cifar | 32 | - | cb | 11,908 | 30 | 59.22 | 59.50 | 1069 |
| `H_ce` | 42 | resnet18-cifar | 32 | - | ce | 11,908 | 18 | 62.54 | 62.44 | 809 |
| `H_decouple` | 42 | resnet18-cifar | 32 | - | decouple | 11,908 | 18 | 71.06 | 71.22 | 679 |
| `H_logit_adjust` | 42 | resnet18-cifar | 32 | - | logit_adjust | 11,908 | 18 | 73.72 | 72.99 | 0 |
| `H_resample` | 42 | resnet18-cifar | 32 | - | resample | 11,908 | 18 | 72.26 | 72.02 | 726 |
| `E1_hog_svm` | – | HOG+SVM (RBF) | 96 | – | – | 8000 | – | 75.00 | – | 199 |
| `E4_ensemble` | 42+43+44 | resnet18 x3 + TTA | 224 | full | lpft | 20,000 | – | 99.22 | – | inference |
| `backbone_ensemble` | 0+1 | convnext_tiny, resnet50, swin_t x2 + TTA | 224 | flip | full | 20,000 | – | 99.70 | – | inference |

Notes:

- The two `E7b` rows are the 8-epoch augmentation check added specifically to
  test whether the 3-epoch null result in Section 7.6 was a budget artefact; the
  two `E7c` rows repeat it at 250 images per class. These four runs were first
  launched by hand; their full arguments are saved in each `metrics.json` and
  are reproduced exactly by the `augcheck` phase of `run_experiments.py`.
- `H_cb` (18 epochs) and `H_cb_long` (30 epochs) are both listed: the second was
  run to give the class-balanced loss a longer schedule before concluding that
  it underperforms (Section 9.4). `H_cb_long` was launched by hand and its
  `metrics.json` does not record the learning rate, so only its epoch count can
  be verified from the saved files.
- `H_logit_adjust` shows `sec = 0` because it is post-hoc: its `metrics.json` is
  written by `ee6483/scripts/eval_logit_adjust.py`, which reuses the `H_ce`
  checkpoint and only changes the decision rule. It has no separate training
  log or checkpoint; its history is copied from `H_ce`.
- The `H_*` runs save the class counts, epochs and per-epoch history in
  `metrics.json`, but not the full argument dictionary that the `E*` runs save;
  their arguments are the defaults of `ee6483/src/train_cifar.py` plus the
  command line in `run_experiments.py`.
- `E4_ensemble` is the evaluation of the three `E4_r18_ft` seeds averaged with
  horizontal-flip TTA: the ResNet-18 reference ensemble used to quantify the
  benefit of seed averaging in Section 5.2. It is not a submission artifact.
- `backbone_ensemble` averages the six top-tier checkpoints of Section 7.8 with
  horizontal-flip TTA: the configuration that produced the final `submission.csv`
  (`scripts/ensemble_predict.py`; details in
  `submissions/submission_backbone_ensemble_ensemble.json`).

## A.1b Backbone comparison runs (Section 7.8)

Produced by the config-driven code base (`src/`, `configs/dogs_cats/backbones/`) on the NTU EEE GPU cluster, one NVIDIA A40 per run, torch 2.11.0+cu128. Logs: `runs/backbones/b_<model>_s<seed>/history.csv`; summary: `results/backbones.csv`. All runs completed 6 epochs.

| Run | Best val acc (%) | Best epoch | Last-epoch val acc (%) | Train time (s) |
|---|---|---|---|---|
| `b_convnext_tiny_s0` | 99.60 | 6 | 99.60 | 264 |
| `b_convnext_tiny_s1` | 99.62 | 6 | 99.62 | 265 |
| `b_efficientnet_b0_s0` | 99.22 | 3 | 99.16 | 249 |
| `b_efficientnet_b0_s1` | 99.26 | 4 | 99.20 | 251 |
| `b_resnet18_s0` | 99.28 | 5 | 99.20 | 175 |
| `b_resnet18_s1` | 99.22 | 5 | 99.22 | 167 |
| `b_resnet34_s0` | 99.32 | 5 | 99.24 | 177 |
| `b_resnet34_s1` | 99.24 | 5 | 99.20 | 179 |
| `b_resnet50_s0` | 99.56 | 5 | 99.52 | 238 |
| `b_resnet50_s1` | 99.58 | 5 | 99.54 | 240 |
| `b_swin_t_s0` | 99.56 | 3 | 99.52 | 296 |
| `b_swin_t_s1` | 99.56 | 6 | 99.56 | 292 |
| `b_vit_b_16_s0` | 99.26 | 6 | 99.26 | 394 |
| `b_vit_b_16_s1` | 99.30 | 6 | 99.30 | 393 |

## A.2 Environment

| Component | Version |
|---|---|
| Python | 3.13.14 |
| PyTorch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| CUDA (torch build) | 12.4 |
| cuDNN | 90100 |
| NumPy | 2.5.2 |
| scikit-learn | 1.9.1 |
| Pillow | 12.3.0 |
| GPU | NVIDIA GeForce RTX 3070 Ti Laptop GPU (8 GB) |
| OS | Windows 11 |

The table lists the versions used for every run in A.1; the backbone runs of
A.1b used torch 2.11.0+cu128 on the cluster. `requirements.txt` gives minimum
versions only, so a fresh install will not reproduce these versions exactly
unless they are installed explicitly. The pretrained weights are torchvision's
`DEFAULT` weights for each architecture (IMAGENET1K_V1 for ResNet-18/34,
IMAGENET1K_V2 for ResNet-50 under the versions above). Within a run, the seed
set {42, 43, 44}, `cudnn.deterministic=True`, `cudnn.benchmark=False` and an
offline-resized image cache (short side 256 px) fix the data order and the
pixels. AMP is the one non-deterministic component, which is why the main
model's headline numbers are means over three seeds rather than single runs.

## A.3 The submitted file

`submission.csv`: first five and last three rows:

```
id,label
1,0
2,0
3,0
4,1
5,1
...
498,0
499,0
500,1
```

Validator output (`ee6483/scripts/validate_submission.py`):

```
file      : submission.csv
rows      : 500
dog(1)    : 259   cat(0): 241
unique ids: 500
PASS: submission.csv is well-formed
```

The validator is run before every upload. It checks the header, the row count,
that the ids are exactly 1…500 with no duplicates, that every label is in
{0, 1}, that the id order matches `sampleSubmission.csv` row by row, and that
the predicted class balance is not degenerate. Class counts cannot detect
predictions permuted among ids or a reversed semantic label convention.
Those require checks of filename-derived numeric ids, the explicit class
mapping, and the per-id prediction records; the CSV validator alone cannot
verify the model's semantic predictions.

## A.4 Dataset integrity summary

From `ee6483/results/data_report.json` (produced by `check_data.py` before any
training):

| Check | Result |
|---|---|
| Files present | 20,000 train + 5,000 val + 500 test |
| Unreadable / corrupt | 0 |
| Test ids | exactly 1…500, all numeric file names |
| Id ↔ `sampleSubmission.csv` | one-to-one match |
| pHash near-duplicates (Hamming ≤ 6) | 40 pairs / 35 + 5 groups: 12 cross the train↔val split (10 distinct validation images), 5 cross the class boundary |
| Placeholder graphics | 6 ("PHOTO UNAVAILABLE" / "Adopted"), labelled inconsistently |
| Mean image size | 400 × 361 px (train), 400 × 361 px (test) |
| Smallest side | 32 px |

## A.5 Reproducing this document

The document is assembled from the section files:

```bash
python -m ee6483.scripts.aggregate          # results/summary.csv
python report/build_report.py               # sections/*.md -> REPORT.md
python report/md_to_docx.py                 # REPORT.md -> EE6483_Group_Report.docx
```

The numbers in the section files are written in directly and were checked by
hand against `ee6483/results/summary.csv`, the per-run `metrics.json` files and
`results/backbones.csv`. They are not substituted automatically, so re-running
an experiment does not update the text: a changed result has to be copied into
the relevant section by hand.
