# Appendix: Complete Experiment Log

Everything reported in the body of this document is derived from the 42 run
directories under `ee6483/results/`. This appendix prints the raw log so that
any number in the main text can be traced to a single command.

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
| `G2_cifar_pretrained` | 42 | resnet18-cifar | 160 | - | ce | 45,000 | 7 | 95.58 | 95.20 | 579 |
| `H_cb` | 42 | resnet18-cifar | 32 | - | cb | 11,908 | 18 | 54.02 | 54.04 | 658 |
| `H_cb_long` | 42 | resnet18-cifar | 32 | - | cb | 11,908 | 30 | 59.22 | 59.50 | 1069 |
| `H_ce` | 42 | resnet18-cifar | 32 | - | ce | 11,908 | 18 | 62.54 | 62.44 | 809 |
| `H_decouple` | 42 | resnet18-cifar | 32 | - | decouple | 11,908 | 18 | 71.06 | 71.22 | 679 |
| `H_logit_adjust` | 42 | resnet18-cifar | 32 | - | logit_adjust | 11,908 | 18 | 73.72 | 72.99 | 0 |
| `H_resample` | 42 | resnet18-cifar | 32 | - | resample | 11,908 | 18 | 72.26 | 72.02 | 726 |
| `E1_hog_svm` | – | HOG+SVM (RBF) | 96 | – | – | 8000 | – | 75.00 | – | 199 |
| `E4_ensemble` | 42+43+44 | resnet18 x3 + TTA | 224 | full | lpft | 20,000 | – | 99.22 | – | inference |

Notes:

- The two `E7b` rows are the 8-epoch augmentation check added specifically to
  test whether the 3-epoch null result in Section 7.6 was a budget artefact.
- `H_cb` (18 epochs) and `H_cb_long` (30 epochs, 2× learning rate) are both
  listed: the second was run to give the class-balanced loss a fair schedule
  before concluding that it underperforms (Section 9.4).
- `H_logit_adjust` shows `sec = 0` because it is post-hoc: it reuses the `H_ce`
  checkpoint and only changes the decision rule.
- `E4_ensemble` is the evaluation of the three `E4_r18_ft` seeds averaged with
  horizontal-flip TTA: the configuration that produced `submission.csv`.

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

Everything is pinned: the same seed set {42, 43, 44}, `cudnn.deterministic=True`,
`cudnn.benchmark=False`, and an offline-resized image cache (short side 256 px)
so that every run sees identical pixels. AMP is the one non-deterministic
component, which is why all headline numbers are means over three seeds rather
than single runs.

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
dog(1)    : 261   cat(0): 239
unique ids: 500
PASS: submission.csv is well-formed
```

The validator is run before every upload. It checks the header, the row count,
that the ids are exactly 1…500 with no duplicates, that every label is in
{0, 1}, that the id order matches `sampleSubmission.csv` row by row, and that
the predicted class balance is not degenerate (which is the symptom of a
reversed label convention or a lexicographic id sort).

## A.4 Dataset integrity summary

From `ee6483/results/data_report.json` (produced by `check_data.py` before any
training):

| Check | Result |
|---|---|
| Files present | 20,000 train + 5,000 val + 500 test |
| Unreadable / corrupt | 0 |
| Test ids | exactly 1…500, all numeric file names |
| Id ↔ `sampleSubmission.csv` | one-to-one match |
| pHash near-duplicates (Hamming ≤ 6) | 40 pairs / 35 + 5 groups: 12 cross the train↔val split, 5 cross the class boundary |
| Placeholder graphics | 6 ("PHOTO UNAVAILABLE" / "Adopted"), labelled inconsistently |
| Mean image size | 400 × 361 px (train), 400 × 361 px (test) |
| Smallest side | 32 px |

## A.5 Reproducing this document

The report itself is generated, not hand-assembled:

```bash
python -m ee6483.scripts.aggregate          # results/summary.csv
python report/make_tokens.py                # results/*/metrics.json -> tokens.json
python report/build_report.py               # sections/*.md -> REPORT.md
python report/md_to_docx.py                 # REPORT.md -> EE6483_Group_Report.docx
```

`build_report.py` substitutes the double-brace placeholders from `tokens.json`, so
re-running the pipeline after new experiments updates every table and every
number quoted in the prose. A stale figure in the text is therefore a bug in the
pipeline, not a transcription error: which is the property we wanted.
