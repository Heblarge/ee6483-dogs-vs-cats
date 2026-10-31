# Dogs vs. Cats: Binary Image Classification with Transfer Learning

EE6483 Artificial Intelligence and Data Mining, Mini Project, Option 2
Group report · 13 November 2026

| Member | Matriculation | Contribution | Questions answered |
|---|---|---|---|
| *name to be filled* | *matric to be filled* | 34% | a(1) to a(4), f |
| *name to be filled* | *matric to be filled* | 33% | a (data), b, c, d |
| *name to be filled* | *matric to be filled* | 33% | e, g, h |

## Abstract

This report studies binary cat/dog classification on 20,000 training images, 5,000 validation images and 500 unlabelled test images. Six classifiers are compared under a single pipeline: hand-crafted HOG features with an SVM, a small CNN trained from scratch, a frozen ImageNet ResNet-18 with a linear head, and fine-tuned ResNet-18, ResNet-34 and ResNet-50 models.

The main model, an ImageNet-pretrained ResNet-18 fine-tuned with the LP-FT schedule, reaches 98.93 ± 0.05% validation accuracy over three seeds. A three-seed ensemble with horizontal-flip test-time augmentation raises this to 99.22% and produces the submitted predictions.

A one-factor-at-a-time ablation ranks the design choices by the size of their effect. Pre-training is the largest factor, worth 7.46 points at a fixed budget. Training-set size is second: accuracy rises from 93.56% with 500 images to 98.12% with 20,000 and saturates beyond 10,000. The depth of fine-tuning (1.88 points) and the input resolution (0.64 points from 160 to 224 pixels) are third-order. Augmentation moves accuracy by less than half a point at all three budgets tested, with the sign depending on the schedule. Backbone size is non-monotone: ResNet-34 falls 1.39 points below ResNet-18, while ResNet-50 rises 0.37 points above it.

An analysis of all 50 validation errors shows that two thirds arise when the animal is not presented as in the training set, either because the face is hidden or because the animal is behind a fence or held in a hand. Grad-CAM attributes correct predictions to the animal's head rather than to the scene. Accuracy falls by 12.7 points under heavy blur, but by only 1.1 points under 30° rotation and 0.4 points under 25% occlusion.

The pipeline is then applied to CIFAR-10, where the ImageNet stem must be replaced by a 3×3 stride-1 stem to avoid destroying a 32-pixel image. Training from scratch reaches 93.60% test accuracy and ImageNet pre-training reaches 95.20%. On a long-tailed CIFAR-10 variant with a 90:1 imbalance ratio, five remedies are compared against plain cross-entropy. Post-hoc logit adjustment performs best, raising test accuracy from 62.44% to 72.99% and macro-F1 from 60.27% to 72.96% without retraining.

Keywords: transfer learning, ResNet, fine-tuning, data augmentation, Grad-CAM, long-tailed recognition, CIFAR-10

---

## Contents

1. Literature Survey
2. Dataset and Pre-processing
3. Model and Training Procedure
4. Hyper-parameter Selection
5. Results
6. Error Analysis
7. Ablation Study
8. Extension to CIFAR-10
9. Class Imbalance
10. Conclusion

References

Appendix: complete experiment log

---

## Declaration

All experiments, results and analysis in this report were produced by the group for this project, using the code in `ee6483/` and the commands listed in Appendix A.5. Where external libraries, pre-trained weights or generative tools were used, they are identified at the point of use; all experimental design, results and interpretation are the group's own. We have read and uphold the NTU Honour Code.

---
