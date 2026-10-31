# 7. Ablation Study

## 7.1 Protocol

This section measures how different choices of model and data processing affect validation accuracy. Answering it with a grid search would confound the factors, so we use a one-factor-at-a-time (OFAT) ablation with a fixed reference configuration:

> reference: ResNet-18, ImageNet pre-trained, 160×160, 3 epochs, full augmentation, all 20,000 training images, seed 42, full fine-tuning.

Each table changes exactly one thing relative to that reference. The short schedule of 3 epochs rather than 8 keeps the 25 ablation runs affordable; the ranking of factors is what the study measures, not the absolute accuracy.

Three caveats apply.

1. 3 epochs under-fits. The reference reaches 98.12% in 3 epochs at 160 px, versus 98.93% in 8 epochs at 224 px, so the task is already close to saturated at this budget. When a model is near the ceiling, most knobs stop mattering: that is itself one of our findings, but it means the small effects below should not be extrapolated to a harder task.
2. Single seed. The seed-to-seed standard deviation of the main model is 0.05 pp over three seeds (98.90, 98.90, 99.00). Differences below 0.3 pp are reported as no measurable effect.
3. The reference is not the final model. The final model uses 224 px and 8 epochs (Section 5). The ablation answers "which choices matter", not "what is the best possible number".

## 7.2 The largest effect: pre-training

This is the only fully controlled comparison in the study: identical architecture, data, augmentation, resolution and optimiser; only the initialisation differs. The from-scratch model was given twice the epochs to compensate.

| Level | Epochs | val acc (%) | Δ |
|---|---|---|---|
| random initialisation | 6 | 90.66 | **−7.46** |
| ImageNet pre-trained (reference) | 3 | 98.12 | 0 |

Starting from ImageNet weights is worth 7.46 accuracy points even after giving the scratch model double the training. Within the 3–6 epoch budget the from-scratch network has learned generic low-level filters but has not yet assembled them into cat/dog detectors; the pre-trained network already has them.

## 7.3 The second largest effect: how much data

| Images per class | Total training images | val acc (%) |
|---|---|---|
| 250 | 500 | 93.56 |
| 1,000 | 2,000 | 96.76 |
| 2,500 | 5,000 | 97.90 |
| 5,000 | 10,000 | 98.20 |
| 10,000 (reference) | 20,000 | 98.12 |

![Learning curve](../ee6483/figures/fig_learning_curve.png)

At 500 images the setting is close to a few-shot problem in the sense of Snell et al. [NeurIPS 2017], except that the backbone is pre-trained rather than meta-learned, which is precisely why 500 images remain workable.

The curve is steep up to ~2,000 images and then flattens hard: the last 10,000 images buy nothing measurable (98.20% at 10,000 images vs 98.12% at 20,000, i.e. −0.08 pp, inside the seed noise), while the first 1,500 images (500 to 2,000) buy 3.20 points and the full range from 500 to 20,000 buys 4.56. Three consequences:

- The curve starts to bend at about 2,000 images: a pre-trained backbone trained on 2,000 images already reaches 96.8%.
- Data quantity and pre-training interact. With 500 images the pre-trained model reaches 93.6%; the from-scratch model at 20,000 images (Section 7.2) reaches 90.7%. A pre-trained model with 2.5% of the data beats a from-scratch model with 100% of it. That is the cleanest possible demonstration of transfer, and it is the answer to the "is the high accuracy just ImageNet knowing cats and dogs?" objection: a category prior cannot explain why 500 images suffice for a *different* architecture, whereas a generic representation can.
- Collecting more than ~10,000 images per class would be wasted effort on this task with this backbone. Effort is better spent on the hard cases identified in Section 6.

## 7.4 Transfer strategy

| Strategy | Trainable parameters | val acc (%) |
|---|---|---|
| Linear probe (head only) | 1,026 | 96.24 |
| Last residual stage + head | 8.4 M | 98.20 |
| Full fine-tune (reference) | 11.18 M | 98.12 |
| LP-FT (1 of 3 epochs as the probe) | 11.18 M | 98.32 |

Unfreezing more of the network helps, but all of the benefit arrives at the first step: the linear probe loses 1.88 points relative to the reference, and going from "last stage" to "all layers" changes nothing measurable. In other words the useful adaptation happens in the last residual block; the early layers of an ImageNet network are already the right feature extractor for photographs of animals. This is the same conclusion as Yosinski et al. [NeurIPS 2014] on layer-wise transferability, reproduced on a different dataset.

LP-FT is the best of the four by a small margin at this budget; its advantage over naive full fine-tuning is clearer at 8 epochs, where the destructive first step on a randomly initialised head costs more (Section 3.4).

The probe length differs between the two settings and this should be read with care. Here the schedule is 3 epochs, so the probe occupies the first of them; the main model (M3) runs for 8 epochs and uses a 2-epoch probe, which is why the row above is not a reproduction of the submitted configuration but a controlled comparison at the ablation budget.

## 7.5 Input resolution

| Resolution | val acc (%) | Relative compute |
|---|---|---|
| 96 | 97.08 | 0.2× |
| 128 | 97.94 | 0.35× |
| 160 (reference) | 98.12 | 1× |
| 224 | 98.76 | 2× |

Accuracy increases monotonically with resolution, by 1.04 pp from 96 to 160 and a further 0.64 pp from 160 to 224. Unlike the augmentation row, this trend is monotone and larger than the noise, so it is a real effect. The mechanism is that at 96 px the discriminative region (face, ears, eye shape) is only a few pixels wide; by 160 px it is resolvable. Doubling compute again for the last 0.6 pp is a poor trade, which is why the main model stops at 224 px (the resolution the pre-trained weights expect) and the ablations use 160 px.

## 7.6 Augmentation strength: a budget-dependent, sub-1 pp effect

| Level | Added transforms | val acc (%) |
|---|---|---|
| none | Resize + CenterCrop only | 98.36 |
| flip | + RandomResizedCrop + HFlip | 98.62 |
| geo | + rotation ±15° | 98.56 |
| color | + ColorJitter | 98.48 |
| full (reference) | + RandAugment + RandomErasing | 98.12 |

At a 3-epoch budget augmentation makes no measurable difference, and the strongest setting is the worst of the five by 0.5 pp. To establish whether this is a property of the transforms or of the budget, the two extremes were re-run at 8 epochs and again with 500 images.

| Regime | augmentation `none` | augmentation `full` | difference |
|---|---|---|---|
| 20,000 images, 3 epochs | 98.36 | 98.12 | **-0.24 pp** |
| 20,000 images, 8 epochs | 98.42 | 98.66 | **+0.24 pp** |
| **500 images, 8 epochs** | 95.94 | 96.30 | **+0.36 pp** |

The third row is the one that matters for the "augmentation substitutes for data" intuition: with only 500 training images the no-augmentation model reaches 100% training accuracy by epoch 8, i.e. it memorises the training set, and yet its validation accuracy is only 0.36 pp below the augmented model's. A pre-trained ResNet-18 on this task simply does not overfit in a way that augmentation can fix.

The sign flips between the two full-data budgets, and the magnitude stays below 0.5 pp in all three cases. Three conclusions follow.

- Augmentation is a sub-1 pp lever on this task, at every budget and data size we tested. The task is saturated by pre-training; there is almost no headroom for regularisation to recover, and the residual errors (Section 6) are label noise and ambiguous viewpoints that no augmentation can fix.
- Its sign depends on the schedule. At 3 epochs the strong transforms (RandAugment + RandomErasing) remove information faster than a short schedule can exploit it, so they cost 0.24 pp; at 8 epochs they have time to pay off and gain 0.24 pp (0.36 pp when data is scarce). All three numbers are within a few tenths of a point and should not be over-interpreted.
- The background-shortcut argument for RandomErasing survives qualitatively. Section 6 shows the model attends to the animal rather than the background, and occlusion robustness is high (a 25% occlusion costs 0.4 pp): which is what RandomErasing is supposed to buy. On this dataset the shortcut was never strong enough to be the limiting factor, so the benefit is small.

The practical implication is that an ablation run at too short a budget can invert the sign of a real effect. The full augmentation policy is retained in the main model because it is neutral to positive at the 8-epoch budget actually used, and because it is what gives the model its robustness to occlusion (Section 6).

## 7.7 The full model ladder

| Model | Budget | val acc (%) | Δ vs M3 |
|---|---|---|---|
| M0 HOG + SVM | 8k images | 75.00 | −23.93 |
| M1 SmallCNN, from scratch | 14 ep @128 | 95.06 | −3.87 |
| M2 ResNet-18 frozen + linear | 6 ep @224 | 97.78 | −1.15 |
| **M3 ResNet-18 LP-FT** | 8 ep @224 | **98.93 ± 0.05** | 0 |
| M4 ResNet-34 LP-FT | 8 ep @224 | 97.54 | -1.39 |
| M5 ResNet-50 LP-FT | 6 ep @160 | 99.30 | +0.37 |

The ladder mixes budgets, so the Δ column is indicative rather than controlled. Two results deserve comment:

- A small CNN trained from scratch reaches 95.06%: much higher than the 80–85% one might expect from the literature. Twenty thousand images is enough for a 1.17M-parameter network to learn this binary task; the task is simply easier than ImageNet-1k.
- Backbone size is non-monotone: ResNet-34 is *worse* than ResNet-18 (97.54% vs 98.93 ± 0.05%, early-stopped at epoch 4) while ResNet-50 is *better* (99.30%, at lower resolution and fewer epochs). The spread across all three sizes is 1.8 points, a quarter of the pre-training effect, and the ordering is not stable. At this dataset size the result depends more on how many epochs each model happens to get than on the architecture, which is exactly why "architecture is second-order" is the defensible claim and "bigger is better/worse" is not.

![All six ablation panels at a glance; red bars are the shared reference run](../ee6483/figures/fig_ablation.png)

## 7.8 Ranked summary

The controlled comparisons are ranked below. Each holds the budget fixed and changes one factor.

| Rank | Factor | Effect on validation accuracy | Mechanism |
|---|---|---|---|
| 1 | Pre-training (ImageNet weights vs random) | +7.46 pp | transferred generic features |
| 2 | Training data, 500 to 20,000 images | +4.56 pp | sample coverage, saturating near 10,000 |
| 3 | Fine-tune depth (head only to last stage) | +1.88 pp | adaptation of the last block |
| 4 | Input resolution, 96 to 224 px | +0.64 pp on top of +1.04 from 96 to 160 | resolvable detail |
| 5 | Augmentation, none to full | -0.24 pp | below 0.5 pp at every budget tested; sign depends on schedule |
| 6 | Backbone size, ResNet-18/34/50 | -1.39 to +0.37 pp | non-monotone; dominated by per-model budget |

One further comparison is not controlled and is therefore excluded from the ranking: moving from hand-crafted HOG features to a frozen deep representation (M0 to M2) is worth 23.93 points. It is the largest gap anywhere in the study, but it changes the features, the classifier and the training procedure at once, so it cannot be attributed to a single factor.

In summary, on this dataset the choice of initialisation and the quantity of training data dominate everything else. The depth of fine-tuning and the input resolution are second-order. Augmentation and backbone size are within noise, or negative, once the task is close to saturated. This ordering is what determined the compute allocation of the project: three seeds of one model, rather than a sweep over backbones.
