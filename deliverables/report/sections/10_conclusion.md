# 10. Conclusion

## 10.1 What we did

We built and evaluated a complete cat/dog classification pipeline: a dataset audit (20,000 / 5,000 / 500 images, no corrupt files, 12 near-duplicate pairs crossing the train/validation split), a cached preprocessing and augmentation pipeline, a ladder of six classifiers from hand-crafted HOG features to a fine-tuned ResNet-50, a controlled one-factor-at-a-time ablation of six factors over 25 runs, an error analysis over all 50 validation errors with Grad-CAM and a corruption test, a transfer of the same code base to CIFAR-10, and a five-method study of class imbalance on a long-tailed CIFAR-10 variant.

## 10.2 What we found

1. The task is close to saturated, and transfer learning is what saturates it. The main model reaches 98.93 ± 0.05% validation accuracy over 3 seeds in eight minutes per run. The gap to the same architecture trained from scratch is 7.46 points, and the gap to hand-crafted features is 23.93 points. Two qualifications apply: a 1.17M-parameter CNN trained from scratch still reaches 95.06%, so the task is easier than the 99% headline suggests; and within the fine-tuned model almost all of the performance already sits in the frozen representation (a linear probe reaches 96.24% at the ablation budget and 97.78% when trained longer), not in the classifier on top of it.
2. The ranking of factors is stable and counter-intuitive. Data and pre-training dominate; architecture and resolution are second-order; most hyper-parameters are third-order. ResNet-50 is +0.37 points above ResNet-18 and ResNet-34 is -1.39 below it: a non-monotone 1.8-point spread that is a quarter of the pre-training effect: and 224 px does not beat 160 px by enough to justify twice the compute. This is the practical content of question (f).
3. The residual errors are not about semantics. Two thirds of the 50 validation errors are images where the animal's face is hidden or the animal is behind a fence or in a hand. The model knows what a cat is; it fails when the presentation differs from the training distribution. That is a data-coverage and augmentation problem, and it is where further work would pay off.
4. The model is texture-driven. Grad-CAM concentrates on the head on correct predictions, occlusion costs almost nothing (RandomErasing trained for it), but blur costs 12.7 points. The evidence hierarchy is face geometry > fur texture > colour/context.
5. The pipeline generalises. Moving to CIFAR-10 (10 classes, 32×32) required changing the stem, the head, the augmentation and the optimiser: and the same conclusion emerged: pre-training beats scratch (95.20% vs 93.60% test accuracy). The pipeline, not the specific model, is the reusable asset.
6. Imbalance is fixable cheaply. On CIFAR-10-LT (ρ=100) the best of five methods lifts test accuracy from 62.44% to 72.99% and macro-F1 from 60.27% to 72.96%, with most of the gain on the rarest classes; post-hoc logit adjustment achieves much of this with no retraining at all.

## 10.3 Limitations

- Validation accuracy is mildly optimistic. It was used for early stopping and for model selection, so it is not an unbiased estimate of generalisation. With 5,000 validation images and ~8 compared configurations the bias is small (order 0.1 pp) but non-zero.
- The test set is unlabelled, so we cannot report test accuracy. Every claim about test images in Section 6 is a manual visual judgement and is labelled as such.
- One seed for most ablations. The ablation matrix uses seed 42 only; differences below ~0.4 pp are within the seed noise we measured on the main model and should not be interpreted as real.
- The ablation uses a 3-epoch budget, which understates the value of regularisation (augmentation, label smoothing) relative to an 8-epoch budget.
- Imbalance study is single-seed, single-ratio. ρ=100 and one seed per method; the ordering of methods is informative, the magnitudes are not.
- Compute constrained the backbone sweep. ResNet-50 was run at 160 px rather than 224 px, and ConvNeXt/ViT were not run at all, because on an 8 GB laptop GPU their cost per epoch is 3–8× higher for a gain that Section 1.4 argues is below the seed noise.

## 10.4 What we would do with more compute

1. Sweep the augmentation policy properly (RandAugment magnitude, crop scale) rather than fixing it: Section 7 shows augmentation is worth less than a point at every budget tested, but it is also what makes the model robust to occlusion (Section 6.6).
2. Train the CIFAR-10 from-scratch model for 200 epochs with cutout/AutoAugment, which is what the published 94–95% figures use, so that the pre-training gap is measured against a properly converged control.
3. Add a ConvNeXt-T or ViT-B/16 fine-tune to test the Section 1.4 claim that architecture has plateaued. Our prediction, based on the literature and on the ResNet-18/34/50 spread, is a gain of <0.5 pp: but that is a prediction, not a measurement.
4. Calibrate and add an open-set rejection head (temperature scaling plus a max-softmax threshold) so that the "no photo available" and animal-free images are rejected instead of being confidently misclassified.
5. Exploit the unlabelled test set with a semi-supervised method such as FixMatch [Sohn et al., NeurIPS 2020], which would require relaxing the rule against using test data for training.

## 10.5 Concluding remarks

On a small, well-curated binary problem, the pre-trained representation and the quantity of data account for most of the achievable accuracy, while architecture and hyper-parameters account for little. The residual errors are dominated by how the data was collected, in particular viewpoint, occlusion and label noise, rather than by how the model was built. Methodologically, fixing the seed, averaging over seeds, validating the submission file against the sample and inspecting all of the errors changed our conclusions more than any change to the model did.

---

## 10.6 Reproducing the numbers

```bash
# environment: Python 3.13, torch 2.6.0+cu124, torchvision 0.21.0, RTX 3070 Ti Laptop (8 GB)
python -m ee6483.src.data.check_data --root data/datasets        # -> results/data_report.json
python -m ee6483.src.data.cache_images --src data/datasets --dst data/cache --short 256
python -m ee6483.scripts.run_experiments --phases main compare ablation lrsearch cifar imbalance
python -m ee6483.scripts.aggregate                                # -> results/summary.{csv,md}
python -m ee6483.src.analyze --which all                          # -> figures/
python -m ee6483.src.analyze_cifar                                # -> CIFAR / imbalance figures
python -m ee6483.scripts.dump_errors                              # -> all 50 validation errors
python report/make_tokens.py && python report/build_report.py     # -> report/REPORT.md
```

The aggregate tables can be regenerated from `ee6483/results/summary.csv`, which is
written by the same pipeline.
