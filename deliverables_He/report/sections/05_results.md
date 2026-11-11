# 5. Results

## 5.1 Validation accuracy of every model

All numbers are accuracy on the 5,000-image validation set. Where a model was trained with more than one seed we report mean ± std over seeds; `n` is the number of seeds.

| ID | Model | Init | Input | Params | val acc (%) | n |
|---|---|---|---|---|---|---|
| M0 | HOG + colour histogram → SVM (RBF) | – | 96 | – | 75.00 | 1 |
| M1 | SmallCNN (4 blocks) | random | 128 | 1.17 M | 95.06 | 1 |
| M2 | ResNet-18, frozen backbone + linear head | ImageNet | 224 | 11.18 M | 97.78 | 1 |
| **M3** | **ResNet-18, LP-FT** | ImageNet | 224 | 11.18 M | **98.93 ± 0.06** | 3 |
| M4 | ResNet-34, LP-FT | ImageNet | 224 | 21.29 M | 97.54 | 1 |
| M5 | ResNet-50, LP-FT | ImageNet | 160 | 23.51 M | 99.30 | 1 |
| M3 ×3 + TTA | ResNet-18 reference ensemble | ImageNet | 224 | 3 × 11.18 M | 99.22 | 3 |
| **Backbone ensemble ×6 + TTA** | **the submitted model** (Section 7.8: ConvNeXt-T, ResNet-50, Swin-T × 2 seeds) | ImageNet | 224 | 6 models, 23.5–27.8 M each | **99.70** | 2 per backbone |

![Model ladder](../ee6483/figures/fig_model_comparison.png)

![Training curves](../ee6483/figures/fig_training_curves.png)

### Observations

1. The ladder spans 23.9 accuracy points (75.00% → 98.93%), and 22.8 of them are crossed in a single step: going from hand-crafted HOG features to a frozen deep representation (M0 → M2). Everything after that, namely unfreezing, more epochs, a bigger backbone and ensembling, competes for the last 1.2 points.
2. Training from scratch is worse, but not catastrophically so at full budget. M1 (1.17M parameters, 14 epochs, 128 px, no pre-training) reaches 95.06%, only 2.7 points below the frozen ImageNet backbone. With 20,000 images, a small CNN can learn this binary task. The scratch-versus-pretrained gap is much larger when the budget is short (Section 7: 90.66% against 98.12% at 3 to 6 epochs) or the data is scarce (Section 7.3). Pre-training therefore substitutes for data and epochs rather than being the only viable approach.
3. Fine-tuning beats freezing by 1.15 points (97.78% → 98.93 ± 0.06%), and Section 4.4 shows that essentially all of that gain comes from unfreezing the last residual block. "Frozen" here means the backbone weights receive no gradient; the BatchNorm layers stay in training mode, so their running statistics still adapt to the new data, and M2 is therefore slightly stronger than a strict fixed-feature linear probe.
4. The backbone-size effect is non-monotone, and small. ResNet-34 reaches 97.54% (-1.39 points vs ResNet-18, early-stopped at epoch 4). That value is its epoch-1 result, before unfreezing: the LP-FT implementation creates a fresh optimiser and schedule when the backbone is unfrozen, which causes a one-epoch validation dip (ResNet-34: 97.52% → 94.58%; M3 seed 42: 97.16% → 94.96%), and with patience 3 the ResNet-34 run stopped before recovering, while ResNet-50 reaches 99.30% (+0.37 points, at 160 px and 6 epochs, i.e. *less* compute per image than M3). So the three sizes span 1.8 points in a non-monotone way: the 34 is an outlier that stopped early, and the 50 is the best single model we trained. Two conclusions follow. First, architecture is not the binding constraint, since the whole 18/34/50 spread is a quarter of the 7.46-point pre-training effect. Second, at this dataset size the result is sensitive to the training budget each model receives, so a single-run backbone comparison carries little information. Section 7.8 therefore repeats it for seven backbones under one fixed budget with two seeds each: ResNet-18 and ResNet-34 are then indistinguishable (99.25% vs 99.28%), and the best backbone, ConvNeXt-T, reaches 99.61 ± 0.01%.
5. Three-seed averaging plus horizontal-flip TTA adds 0.29 points (98.93% → 99.22%) and reduces the error count from 50 to 39. It requires no retraining and no new data, only variance reduction. The gain is just below the 0.3 pp threshold that Section 7.1 applies to single-seed comparisons; it rests on three seeds and an 11-error reduction rather than on one run, but it is still a small effect.

M0 (HOG+SVM) is the exception to the 5,000-image validation protocol: it was trained on 8,000 and validated on 2,000 images because feature extraction is CPU-bound; all other rows use the full 5,000-image validation set. The single ResNet-50 run reaches 99.30%, which is 0.08 points above the M3 reference ensemble (99.22%) but 0.40 points below the submitted backbone ensemble (99.70%). It is one seed, trained at 160 px for 6 epochs, and the seed-to-seed standard deviation of the ResNet-18 family is 0.06, so a 0.08-point difference between one run of one model and a three-seed ensemble of another cannot be separated from run-to-run variation. Making the comparison meaningful would need three ResNet-50 seeds of its own (about one hour of GPU time) plus a decision about which resolution to use for all three. This comparison was later made properly in Section 7.8, which retrains seven backbones under one budget with two seeds each; the submitted model is built from its top tier (Section 5.3).
6. The seed spread of the main model is tiny (98.90, 98.90, 99.00 → sample standard deviation 0.06 pp; every ± in this report is a sample standard deviation, ddof = 1). At this point the task is approaching its label-noise ceiling: Section 6 attributes 5 of the 50 single-model errors to label noise, non-photographic content or severely degraded images. At least three of them (Section 6, category 5) are mislabelled or contradictory, so a sound classifier should not be expected to get them right; this suggests a practical ceiling in the region of 99.9% on this validation set, as an estimate rather than a hard bound.

## 5.2 Detailed metrics for the ResNet-18 reference ensemble (M3 ×3 + TTA)

The metrics below quantify seed averaging for the controlled ResNet-18 study. Sections 6.2–6.6 use one M3 seed for model-level interpretability, while Sections 6.7–6.8 use the submitted ensemble for test-set inspection. The submitted backbone ensemble reaches 99.70% (15 errors out of 5,000); see Section 5.3.

| Metric | Value |
|---|---|
| Validation accuracy (3-seed ensemble + TTA) | **99.22%** |
| Errors | 39 / 5,000 |
| Macro-F1 | 99.22% (identical to accuracy: the classes are exactly balanced) |
| ROC-AUC | 0.9996 |
| Confusion matrix | cat→dog 17, dog→cat 22 |

Because the validation set is exactly 2,500/2,500, accuracy, balanced accuracy and macro-F1 coincide up to rounding: there is no class-prior artefact to correct for. This is *not* true in Section 9, which is why all three metrics are reported there.

The confusion matrix of the *single-seed* model (the one used for the error analysis in Section 6) is shown in Section 6.2 rather than repeated here; its off-diagonal counts are 18 cat→dog and 32 dog→cat, and the ensemble reduces the total from 50 errors to 39.

## 5.3 Test set and submission file

The submitted `submission.csv` is produced by `scripts/ensemble_predict.py` (cluster job `cluster/ensemble.sbatch`):

1. loading the six top-tier checkpoints of Section 7.8 (ConvNeXt-T, ResNet-50 and Swin-T, seeds 0 and 1 each),
2. applying horizontal-flip test-time augmentation inside each model,
3. averaging the six softmax outputs.

The three backbone families were selected after the controlled comparison in Section 7.8, as the top three by validation accuracy (two seeds each); equal weighting and horizontal-flip TTA were fixed in advance. Two smaller ensembles were evaluated alongside it for reference only.

| Ensemble | Models | val acc (%) | Errors / 5,000 |
|---|---|---|---|
| ResNet-18 × 2 seeds (reference) | 2 | 99.30 | 35 |
| ResNet-18 × 2 seeds + TTA (reference) | 2 | 99.28 | 36 |
| ConvNeXt-T × 2 seeds (reference) | 2 | 99.66 | 17 |
| ConvNeXt-T × 2 seeds + TTA (reference) | 2 | 99.64 | 18 |
| Top tier × 2 seeds | 6 | 99.70 | 15 |
| **Top tier × 2 seeds + TTA (submitted)** | 6 | **99.70** | **15** |

Ensembling six top-tier models removes 24 of the 39 validation errors of the M3 reference ensemble. Horizontal-flip TTA no longer helps once the models are this strong (±1 error), and is kept only because it was part of the pre-registered configuration.

| Property | Value |
|---|---|
| Rows (excluding header) | 500 |
| Columns | `id,label` |
| ids | 1 … 500, one per test image, matching the file names exactly |
| Label convention | 1 = dog, 0 = cat |
| Predicted dogs | 259 |
| Predicted cats | 241 |
| Mean max-softmax confidence | 0.95 |

Sanity checks performed (CSV format validator plus a separate per-id probability comparison):

- header exactly `id,label`; 500 data rows; ids exactly `1..500` with no duplicates;
- every label in `{0,1}`; the id order matches `sampleSubmission.csv` row by row;
- prediction ids and labels agree row by row between `submission.csv` and `submission_backbone_ensemble_probs.csv`; each label agrees with the probability decision under cat=0, dog=1. The loader derives ids from image filenames and checks the training class mapping. The 259/241 class count is descriptive only: it cannot detect predictions assigned to the wrong ids or verify the semantic correctness of labels.

The validator is part of the repository and must pass before any upload.

No test accuracy is reported, because the test set has no labels. We also did not use the test set for model selection, ensembling weights or pseudo-labelling. The validation accuracy above is the only performance number we can defend.

## 5.4 A note on the optimism of validation accuracy

Validation accuracy was used for early stopping and model selection, so the reported results are subject to selection optimism. The reference ensemble's 99.22% and submitted ensemble's 99.70% were measured on the same validation set used for selection; choosing the top three backbone families adds another selection step. Without an independent labelled holdout or a nested evaluation, we cannot quantify the size of this bias or claim a numerical upper bound. The unlabelled test set cannot resolve this uncertainty.

## 5.5 Runtime

| Experiment | Time on RTX 3070 Ti Laptop (8 GB, mixed precision) |
|---|---|
| ResNet-18, 8 epochs, 224 px, 20k images | ~8 min |
| ResNet-34, 8 epochs, 224 px | ~11 min |
| ResNet-50, 6 epochs, 160 px | ~5 min |
| SmallCNN, 14 epochs, 128 px | ~8 min |
| Frozen backbone, 6 epochs, 224 px | ~6 min |
| Ablation run (3 epochs, 160 px) | ~2 min |
| CIFAR-10 G1 (25 epochs, 45k images) | ~26 min |

Mixed precision roughly halves all of the above relative to fp32. The full experiment matrix, comprising 3 main seeds, 5 other ladder models, 25 ablation runs, 2 CIFAR-10 models and 6 imbalance runs, takes about 3.7 hours of training time on this single GPU (the sum of the per-run times in Appendix A.1).
