# 2. Dataset and Pre-processing

## 2.1 Amount of data used

| Split | cat | dog | total | labels | used for |
|---|---|---|---|---|---|
| train | 10,000 | 10,000 | **20,000** | Yes | gradient updates |
| val | 2,500 | 2,500 | **5,000** | Yes | model selection, early stopping, all reported accuracies |
| test | – | – | **500** | No | `submission.csv` only |

What each model consumed

| Model | train images | val images | note |
|---|---|---|---|
| M0 HOG+SVM | 8,000 (4,000/class) | 2,000 (1,000/class) | CPU feature extraction is the bottleneck |
| M1 SmallCNN | 20,000 | 5,000 | full data |
| M2 ResNet-18 frozen | 20,000 | 5,000 | backbone not updated |
| **M3 ResNet-18 LP-FT** | **20,000** | **5,000** | main model |
| M4 ResNet-34 | 20,000 | 5,000 | |
| M5 ResNet-50 | 20,000 | 5,000 | 160 px to fit the 8 GB budget |
| Section 7 data-quantity sweep | 500 / 2,000 / 5,000 / 10,000 / 20,000 | 5,000 | nested class-balanced subsets |

The nested subsets are drawn with a fixed permutation (seed 42) per class, so the 500-image set is a strict subset of the 2,000-image set, and so on. Without nesting the learning curve would mix "more data" with "different data".

Discipline. Validation data never contributes to a gradient step, and the test set is never used for model selection, hyper-parameter tuning or pseudo-labelling. Validation accuracy is used for early stopping and for choosing among the M0–M5 ladder; this introduces a mild optimistic bias in the reported validation numbers, which we state explicitly in Section 5 instead of hiding it.

## 2.2 Dataset health check

Run before any training (`ee6483/src/data/check_data.py`); the raw output is `ee6483/results/data_report.json`.

| Check | Result |
|---|---|
| Image counts | exactly 10,000 / 2,500 per class; 500 test images |
| Corrupt / unreadable files | **0** in all splits |
| Colour mode | all RGB after `convert("RGB")` |
| Width (train) | cat 50–1023 (mean 411), dog 42–1050 (mean 398) |
| Height (train) | cat 32–768 (mean 357), dog 33–702 (mean 365) |
| Mean aspect ratio | 1.19 (cat), 1.13 (dog): mild landscape bias |
| Smallest side | **32 px** (train), 37 px (test) → a handful of images must be upscaled |
| Test file names | all numeric, ids exactly 1…500, contiguous |
| `sampleSubmission.csv` | 500 rows, header `id,label`, ids match the test files one-to-one |
| Near-duplicates (pHash, Hamming ≤ 6) | **40 pairs** in 35 + 5 groups: 28 identical hashes, the rest at distance 2 |
| … crossing the train↔val split | **12 pairs** (0.24% of the validation set) |
| … crossing the class boundary | **5 pairs** |
| Placeholder graphics | **6 images** ("PHOTO UNAVAILABLE" / "Adopted"), 4 of them the same graphic labelled both cat and dog |

Three consequences:

1. Resolution heterogeneity is real. The 32 px minimum is well below the 224 px input, so those images are upsampled by a factor of 7. They are a known source of error in Section 6. Caching at short side 256 px (Section 2.3) normalises the rest.
2. The dataset contains placeholder graphics, and they are labelled inconsistently. Six images are not photographs at all: four copies of a "PHOTO UNAVAILABLE" black-cat silhouette and two of an "Adopted" text card. The "PHOTO UNAVAILABLE" graphic appears three times in the training set under `dog` and once in the validation set under `cat`; the "Adopted" card appears once under each label, both in training. Only one of the six therefore lies in the validation set, and that image is unlearnable by construction: the network sees the same pixels labelled `dog` three times during training and is then asked for `cat`. One unavoidable error in 5,000 images caps validation accuracy at 1 - 1/5,000 = 99.98%. The two training-set copies of the "Adopted" card cost nothing directly; they simply contribute contradictory gradients. We did not delete them: removing six images would not change any conclusion, and keeping them makes the label-noise discussion in Section 6 concrete.
3. Twelve near-duplicate pairs cross the train/val split (0.24% of the 5,000 validation images). The largest effect this can have is 0.24 pp of validation accuracy, which is larger than the seed-to-seed spread of the main model (0.05 pp) but an order of magnitude below any effect we report. We did not remove them either: the split was given, and quietly cleaning the validation set would make our numbers incomparable with anyone else's on the same data. The pairs are listed in `ee6483/results/duplicates_all.json` so the reader can re-run the analysis with them excluded.

![Near-duplicates and placeholders](../ee6483/figures/fig_duplicates.png)

## 2.3 Pre-processing

```
decode → convert("RGB") → cache resize (short side 256, bicubic, JPEG q=92)
       → Resize(256) → CenterCrop(224) → ToTensor → Normalize(μ, σ)
```

- RGB conversion: a few images are grayscale or CMYK; forcing RGB keeps the channel count uniform.
- Short-side-256 cache. Decoding 20,000 JPEGs at every epoch is the dataloader bottleneck (~400×400 average). Resizing once to short side 256 and storing as JPEG q=92 cuts the epoch time roughly in half and, more importantly, guarantees that *every experiment sees identical pixels* regardless of when it was run. Images already smaller than 256 px are not upscaled in the cache.
- Resize(256) → CenterCrop(224) at evaluation time is deterministic, so validation accuracy is comparable across runs. The 256/224 ratio (1.14) preserves the ImageNet convention that the model was pre-trained with.
- ImageNet normalisation $\mu=(0.485,0.456,0.406)$, $\sigma=(0.229,0.224,0.225)$. This is not optional: the pre-trained weights expect inputs with these statistics, and using dataset-specific statistics instead costs several points of accuracy.

## 2.4 Augmentation

Scaling, rotation and flipping are all used, together with colour and occlusion perturbations.

| Level | RandomResizedCrop | HFlip | Rotation | ColorJitter | RandAugment | RandomErasing |
|---|---|---|---|---|---|---|
| `none` | – (Resize+CenterCrop) | – | – | – | – | – |
| `flip` | Yes (0.5–1.0) | Yes | – | – | – | – |
| `geo` | Yes | Yes | ±15° | – | – | – |
| `color` | Yes | Yes | ±15° | Yes | – | – |
| `full` | Yes | Yes | ±15° | Yes | Yes (2 ops, m=7) | Yes (p=0.25) |

Rationale, per transform:

- RandomResizedCrop(224, scale=(0.5, 1.0)): provides scale *and* translation jitter in one op, and the 0.5 lower bound forces the model to recognise animals that occupy only half the frame. It also crops away background, which is the single most useful defence against the "dog on grass" shortcut.
- Horizontal flip: the only label-preserving geometric symmetry for animals. Free accuracy, no risk.
- Rotation ±15°: animals appear at arbitrary in-plane angles; ±15° covers most of the realistic range without introducing black corners large enough to act as a cue.
- No vertical flip. Upside-down animals are out of distribution; adding them teaches an invariance that never helps at test time and slightly hurts (we verified this informally: vertical flip consistently cost 0.2–0.5 pp, consistent with the "augmentation must match the test distribution" principle).
- ColorJitter(0.2, 0.2, 0.2, 0.05): indoor/outdoor lighting differs enormously between the two classes, so colour jitter also weakens a spurious cue.
- RandAugment (2 ops, magnitude 7): replaces hand-tuned policy search with a single magnitude parameter; magnitude 7 of 30 is moderate, appropriate for a 20k-image dataset where over-regularisation is a real risk.
- RandomErasing (p=0.25): occludes a random rectangle so the model cannot rely on a single discriminative part (an ear, a collar). It is the augmentation most directly aimed at the background/part shortcut.
- mixup / CutMix are implemented and configurable (`--aug mix`) but not used for the main model: with 8 epochs and label smoothing already active, the extra regularisation did not help in our runs, and mixup makes the training curves harder to interpret for a report of this length.

## 2.5 Illustrations

![Eight training images per class: breed, pose, scale and background all vary](../ee6483/figures/fig_dataset_samples.png)

![One training image passed through the full augmentation pipeline six times](../ee6483/figures/fig_augmentation.png)
