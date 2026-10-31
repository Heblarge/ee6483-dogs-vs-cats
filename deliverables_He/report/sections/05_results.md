# 5. Results

## 5.1 Validation accuracy of every model

All numbers are accuracy on the 5,000-image validation set. Where a model was trained with more than one seed we report mean ± std over seeds; `n` is the number of seeds.

| ID | Model | Init | Input | Params | val acc (%) | n |
|---|---|---|---|---|---|---|
| M0 | HOG + colour histogram → SVM (RBF) | – | 96 | – | 75.00 | 1 |
| M1 | SmallCNN (4 blocks) | random | 128 | 1.17 M | 95.06 | 1 |
| M2 | ResNet-18, frozen backbone + linear head | ImageNet | 224 | 11.18 M | 97.78 | 1 |
| **M3** | **ResNet-18, LP-FT** | ImageNet | 224 | 11.18 M | **98.93 ± 0.05** | 3 |
| M4 | ResNet-34, LP-FT | ImageNet | 224 | 21.29 M | 97.54 | 1 |
| M5 | ResNet-50, LP-FT | ImageNet | 160 | 23.51 M | 99.30 | 1 |
| **M3 ×3 + TTA** | **the submitted model** | ImageNet | 224 | 3 × 11.18 M | **99.22** | 3 |

![Model ladder](../ee6483/figures/fig_model_comparison.png)

![Training curves](../ee6483/figures/fig_training_curves.png)

### Observations

1. The ladder spans 23.9 accuracy points (75.00% → 98.93%), and 22.8 of them are crossed in a single step: going from hand-crafted HOG features to a frozen deep representation (M0 → M2). Everything after that, namely unfreezing, more epochs, a bigger backbone and ensembling, competes for the last 1.2 points.
2. Training from scratch is worse, but not catastrophically so at full budget. M1 (1.17M parameters, 14 epochs, 128 px, no pre-training) reaches 95.06%, only 2.7 points below the frozen ImageNet backbone. With 20,000 images, a small CNN can learn this binary task. The scratch-versus-pretrained gap is much larger when the budget is short (Section 7: 90.66% against 98.12% at 3 to 6 epochs) or the data is scarce (Section 7.3). Pre-training therefore substitutes for data and epochs rather than being the only viable approach.
3. Fine-tuning beats freezing by 1.15 points (97.78% → 98.93 ± 0.05%), and Section 4.4 shows that essentially all of that gain comes from unfreezing the last residual block.
4. The backbone-size effect is non-monotone, and small. ResNet-34 reaches 97.54% (-1.39 points vs ResNet-18, early-stopped at epoch 4), while ResNet-50 reaches 99.30% (+0.37 points, at 160 px and 6 epochs, i.e. *less* compute per image than M3). So the three sizes span 1.8 points in a non-monotone way: the 34 is an outlier that stopped early, and the 50 is the best single model we trained. Two conclusions follow. First, architecture is not the binding constraint, since the whole 18/34/50 spread is a quarter of the 7.46-point pre-training effect. Second, at this dataset size the result is sensitive to the training budget each model receives, so a single-run backbone comparison carries little information. Section 7.8 therefore repeats it for seven backbones under one fixed budget with two seeds each: ResNet-18 and ResNet-34 are then indistinguishable (99.25% vs 99.28%), and the best backbone, ConvNeXt-T, reaches 99.61 ± 0.01%.
5. Three-seed averaging plus horizontal-flip TTA adds 0.29 points (98.93% → 99.22%) and reduces the error count from 50 to 39. This is the cheapest accuracy in the project, requiring no retraining and no new data, only variance reduction, which is why it is what we submit.

The single ResNet-50 run reaches 99.30%, which is 0.08 points above the submitted ensemble. It is one seed, trained at 160 px for 6 epochs, and the seed-to-seed standard deviation of the ResNet-18 family is 0.05, so a 0.08-point difference between one run of one model and a three-seed ensemble of another cannot be separated from run-to-run variation. Making the comparison meaningful would need three ResNet-50 seeds of its own (about one hour of GPU time) plus a decision about which resolution to use for all three. The ensemble was retained because it is the configuration whose variance is measured rather than assumed.
6. The seed spread of the main model is tiny (98.90, 98.90, 99.00 → σ = 0.05 pp). At this point the task is close to its label-noise ceiling: Section 6 shows that at least 5 of the 50 single-model errors are unlearnable, so ~99.9% is the practical maximum on this validation set.

## 5.2 Detailed metrics for the submitted model

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

The submitted `submission.csv` is produced by:

1. loading the three seed-42/43/44 checkpoints of M3,
2. applying horizontal-flip test-time augmentation inside each model,
3. averaging the three softmax outputs.

| Property | Value |
|---|---|
| Rows (excluding header) | 500 |
| Columns | `id,label` |
| ids | 1 … 500, one per test image, matching the file names exactly |
| Label convention | 1 = dog, 0 = cat |
| Predicted dogs | 261 |
| Predicted cats | 239 |
| Mean max-softmax confidence | 0.94 |

Sanity checks performed (`ee6483/scripts/validate_submission.py`):

- header exactly `id,label`; 500 data rows; ids exactly `1..500` with no duplicates;
- every label in `{0,1}`; the id order matches `sampleSubmission.csv` row by row;
- the class balance is 52/48, consistent with a balanced test set: a strong signal that the id→file mapping and the label convention are both correct. A reversed convention or a lexicographic id sort would have produced a very different (or degenerate) distribution.

The validator is part of the repository and must pass before any upload.

No test accuracy is reported, because the test set has no labels. We also did not use the test set for model selection, ensembling weights or pseudo-labelling. The validation accuracy above is the only performance number we can defend.

## 5.4 A note on the optimism of validation accuracy

Validation accuracy was used for early stopping and for choosing among M0–M5, so the reported numbers are mildly optimistic relative to a model that had never seen the validation set. The bias is small, with 5,000 validation images and about 10 compared configurations, but it is real. The appropriate statement is that *99.22% is a validation-set number obtained with model selection performed on the same set; the test set is unlabelled, so we cannot quantify the gap.*

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

Mixed precision roughly halves all of the above relative to fp32. The full experiment matrix, comprising 3 main seeds, 6 ladder models, 25 ablation runs, 2 CIFAR-10 models and 6 imbalance runs, takes about 3.5 hours end-to-end on this single GPU.
