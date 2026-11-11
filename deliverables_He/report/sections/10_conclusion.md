# 10. Conclusion

## 10.1 What we did

We built and evaluated a complete cat/dog classification pipeline: a dataset audit (20,000 / 5,000 / 500 images, no corrupt files, 12 near-duplicate pairs crossing the train/validation split), a cached preprocessing and augmentation pipeline, a ladder of six classifiers from hand-crafted HOG features to a fine-tuned ResNet-50, a controlled one-factor-at-a-time ablation of six factors over 25 runs, a fixed-budget comparison of seven ImageNet backbones over 14 runs whose top tier forms the submitted ensemble (99.70% validation accuracy), an error analysis over all 50 validation errors with Grad-CAM and a corruption test, a transfer of the same code base to CIFAR-10, and a five-method study of class imbalance on a long-tailed CIFAR-10 variant.

## 10.2 What we found

1. The task is close to saturated, and transfer learning is what saturates it. The main model reaches 98.93 ± 0.05% validation accuracy over 3 seeds in eight minutes per run. The gap to the same architecture trained from scratch is 7.46 points, and the gap to hand-crafted features is 23.93 points. Two qualifications apply: a 1.17M-parameter CNN trained from scratch still reaches 95.06%, so the task is easier than the 99% headline suggests; and within the fine-tuned model almost all of the performance already sits in the frozen representation (a linear probe reaches 96.24% at the ablation budget and 97.78% when trained longer), not in the classifier on top of it.
2. The ranking of factors is stable and counter-intuitive. Data and pre-training dominate; architecture and resolution are second-order; most hyper-parameters are third-order. In the single-run ladder ResNet-34 fell 1.39 points below ResNet-18, but under a fixed budget the two are indistinguishable, and seven backbones span only 0.37 points (99.24% to 99.61%), ordered by the ImageNet accuracy of their weights rather than by size, with the under-trained ViT-B/16 as the exception. 224 px does not beat 160 px by enough to justify twice the compute. This is the practical content of question (f).
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
- The backbone sweep (Section 7.8) uses two seeds per model and a single 6-epoch recipe tuned for CNN fine-tuning, which may disadvantage ViT-B/16; it was run with a separate code base, so its numbers are not directly comparable with the main model's.

## 10.4 What we would do with more compute

1. Sweep the augmentation policy properly (RandAugment magnitude, crop scale) rather than fixing it: Section 7 shows augmentation is worth less than a point at every budget tested, but it is also what makes the model robust to occlusion (Section 6.6).
2. Train the CIFAR-10 from-scratch model for 200 epochs with cutout/AutoAugment, which is what the published 94–95% figures use, so that the pre-training gap is measured against a properly converged control.
3. Fine-tune ViT-B/16 for longer than 6 epochs: Section 7.8 shows it still improving at the end of the fixed budget, so its ceiling on this task is not yet measured. The same sweep confirmed the prediction made for ConvNeXt-T, a gain below 0.5 pp (0.36 pp over ResNet-18).
4. Calibrate and add an open-set rejection head (temperature scaling plus a max-softmax threshold) so that the "no photo available" and animal-free images are rejected instead of being confidently misclassified.
5. Exploit additional unlabelled data (not the test set, which must stay untouched) with a semi-supervised method such as FixMatch [Sohn et al., NeurIPS 2020]. Applying it to the test images would violate the evaluation protocol and should not be done.

## 10.5 Concluding remarks

On a small, well-curated binary problem, the pre-trained representation and the quantity of data account for most of the achievable accuracy, while architecture and hyper-parameters account for little. The residual errors are dominated by how the data was collected, in particular viewpoint, occlusion and label noise, rather than by how the model was built. Methodologically, fixing the seed, averaging over seeds, validating the submission file against the sample and inspecting all of the errors changed our conclusions more than any change to the model did.

---

## 10.6 Reproducing the numbers

```bash
# Sections 2-9 (ee6483/): Python 3.13, torch 2.6.0+cu124, torchvision 0.21.0, RTX 3070 Ti Laptop (8 GB)
python -m ee6483.src.data.check_data --root ../datasets/datasets   # -> results/data_report.json
python -m ee6483.src.data.cache_images --src ../datasets/datasets --dst data/cache --short 256
mkdir -p data/cifar
curl -fL -o data/cifar/train.parquet https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/train-00000-of-00001.parquet
curl -fL -o data/cifar/test.parquet https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet
python -m ee6483.scripts.run_experiments --phases main compare ablation augcheck lrsearch extra cifar imbalance
python -m ee6483.scripts.aggregate                                # -> results/summary.{csv,md}
python -m ee6483.src.analyze --which all                          # -> figures/
python -m ee6483.src.analyze_cifar                                # -> CIFAR / imbalance figures
python -m ee6483.scripts.dump_errors                              # -> all 50 validation errors

# Section 7.8 and the final submission (src/): torch 2.11.0+cu128, one A40 per run
bash scripts/run_backbones.sh                                     # 7 backbones x seeds 0, 1
ckpts() { printf 'runs/backbones/b_%s_s0/checkpoints/best.pt,runs/backbones/b_%s_s1/checkpoints/best.pt' "$1" "$1"; }
python scripts/ensemble_predict.py --out submissions/submission_backbone_ensemble.csv \
  --group convnext="$(ckpts convnext_tiny)" --group resnet18="$(ckpts resnet18)" \
  --group top3="$(ckpts convnext_tiny),$(ckpts resnet50),$(ckpts swin_t)" --choose top3 --tta
cp submissions/submission_backbone_ensemble.csv submissions/submission.csv

python report/build_report.py && python report/md_to_docx.py      # -> report/REPORT.md, .docx
```

`run_experiments.py` reproduces every `ee6483/` run in Appendix A.1 except the
learning rate of `H_cb_long`, which was not saved (Appendix A.1, notes). The exact
checkpoint lists for the ensemble are in
`submissions/submission_backbone_ensemble_ensemble.json`. AMP makes individual runs
non-bit-exact, so re-runs should match to within the seed noise reported in the text,
not to the last digit. The numbers in the section files are maintained by hand
(Appendix A.5).

Tracked metrics without their git-ignored checkpoints are treated as incomplete and
are retrained automatically, overwriting the tracked files, so reproductions should be run
in a separate checkout. The recipes, prescribed seeds and six-model ensemble are
frozen for reproduction: reruns must not be cherry-picked or retuned after observing
validation results. Validation is used for early stopping/model selection only; the
dog/cat test set remains prediction-only, and logit-adjustment tau is selected on the
CIFAR validation split before a single test evaluation. This prevents direct leakage,
although the residual validation-selection bias noted in Section 10.3 cannot be made
exactly zero without a new held-out set.
