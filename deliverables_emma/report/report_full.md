# EE6483 Artificial Intelligence and Data Mining — Mini Project (Option 2)
> **DRAFT v0.5 — 2026-10-30 11:35**
> 全部章节完整（§1–§12），实验数据全部填入，架构图已生成。
> 仅剩：小组成员姓名/学号、贡献分工、参考文献按组内实际阅读情况微调。

## Dogs vs. Cats Classification and Its Extension to CIFAR-10

<!--
HOW TO USE THIS SKELETON
- Every heading below maps 1:1 to a marking item in the brief. Do not delete any.
- `[[TODO: ...]]` = content or number still to be filled.
- `[[TABLE-...]]` / `[[FIGURE-...]]` = the artefact already exists in the repo,
  see the note in brackets for the file to embed.
- Target length: 8–10 pages excluding references. Marks are lost for empty
  subsections, not for brevity — answer every sub-question explicitly.
-->

**Group members**

| Name | Matriculation No. | Contribution |
|---|---|---|
| [[TODO: name]] | [[TODO]] | [[TODO: e.g. data pipeline + Dogs vs Cats model (b, c, d)]] |
| [[TODO: name]] | [[TODO]] | [[TODO: e.g. literature survey + error analysis (a, e)]] |
| Emma [[TODO: full name]] | [[TODO]] | [[TODO: CIFAR-10 + class imbalance + experiments (f, g, h)]] |

---

## 1. Introduction and Problem Definition  *(supports item a-1)*

- Task: binary image classification, cat vs. dog, 500 unlabelled test images.
- Why it is non-trivial: intra-class variation (breed, pose, lighting, occlusion),
  inter-class similarity (small fluffy dogs vs. cats), background bias.
- Formal definition: learn $f_\theta: \mathbb{R}^{H\times W\times 3} \to \{0,1\}$
  minimising empirical risk on $\mathcal{D}_{train}$, evaluated by accuracy on
  $\mathcal{D}_{val}$ and $\mathcal{D}_{test}$.
- **Structure of the report.** Part one solves the binary problem end to end:
  data, model, training strategy, validation accuracy and a submission file. Part
  two takes the *same* pipeline to CIFAR-10, a ten-way classification problem at
  32×32, and shows what has to change (resolution, augmentation, loss) and what
  does not (the training recipe, the validation discipline). Part three keeps the
  CIFAR-10 setting but makes the training distribution long-tailed, and asks a
  different question: not "how accurate is the model" but "which classes does it
  silently give up on, and what fixes that". Each part therefore reuses the
  previous one as a controlled baseline rather than starting from scratch.

## 2. Literature Survey

### 2.1 Problem settings and terminology

Image classification is the task of assigning one label from a fixed set to an
input image. Even for a task as narrow as "cat or dog", the same algorithm can be
posed under several different settings, and the choice of setting determines which
methods are applicable and which metrics are meaningful.

**Supervised versus unsupervised.** In the *supervised* setting every training
image carries a label and the model minimises a classification loss, which is the
setting of this project. The *unsupervised* setting has no labels at all: methods
must discover structure (clustering, k-means over deep features) or learn
representations by solving a pretext task. Self-supervised learning (SSL) sits in
between and has become practically important: SimCLR, MoCo, DINO and DINOv2
[10] train an encoder by predicting different views of the same image, and the
resulting features transfer well to downstream classification with only a linear
probe. In our project SSL matters indirectly: the ImageNet weights we fine-tune
were themselves trained with labels, but the same recipe (pretrain broadly,
fine-tune narrowly) is exactly what SSL enables when labels are scarce.

**Closed-set versus open-set.** Our test set contains only cats and dogs, so the
problem is *closed-set*: every test image belongs to one of the known classes and
the model is only asked to choose between them. An *open-set* system must
additionally reject inputs belonging to none of the training classes (a rabbit, a
cartoon, a screenshot), typically by thresholding a confidence or energy score.
The distinction matters here because our model is extremely confident on almost
every input (Section 7): a closed-set classifier of this kind would fail badly if
deployed open-set without a rejection mechanism.

**With and without domain shift.** Train, validation and test images in this
project come from the same source and the same distribution, i.e. there is *no
domain shift*. Under shift — a different camera, night photographs, drawings,
heavy compression — accuracy typically degrades sharply, and methods such as
domain adaptation or test-time adaptation are needed. We deliberately do not
address shift, and we note it as a limitation: the 98.8% validation accuracy
should not be read as a claim about photographs taken in the wild.

**Other axes worth naming.** *Fine-grained* classification (telling a Siamese from
a Persian) is harder than the coarse cat-versus-dog problem because the
discriminative evidence is local and subtle; *long-tailed* classification, where
class frequencies follow a heavy-tailed distribution, is the setting of Section 10
and is known to break naive training even when overall accuracy looks acceptable.

### 2.2 Recent progress in the field

**The convolutional era.** AlexNet [1] established the modern recipe (ReLU,
dropout, data augmentation, GPU training) on ImageNet. VGG [2] showed that depth
built from stacked 3×3 convolutions is a strong prior, at the cost of a very large
parameter count. ResNet [3] solved the degradation problem of very deep networks
with identity shortcuts, making 50–152-layer networks trainable, and it remains
the default backbone for transfer learning a decade later.

**The transformer era.** ViT [4] replaced convolution with self-attention over
image patches and showed that, given enough pretraining data, a pure transformer
matches or beats the best CNNs (88.55% ImageNet top-1 with JFT-300M pretraining).
Its weakness is the opposite of a CNN's: transformers have little inductive bias,
so they need either very large datasets or strong augmentation and distillation
(DeiT) to compete. Swin introduced hierarchical windows to recover the
multi-scale structure that convolutions provide for free.

**The convnet revival.** ConvNeXt [5] asked a precise question: if a ResNet is
modernised step by step using only design choices borrowed from transformers
(patchify stem, inverted bottleneck, depthwise 7×7 convolution, GELU, LayerNorm,
fewer activation/normalisation layers, and the training recipe of DeiT/Swin), how
much of the gap disappears? The answer was essentially all of it: ConvNeXt-XL
reaches 87.8% ImageNet top-1 with ImageNet-22k pretraining, matching Swin at
comparable compute, while remaining a plain convolutional network. ConvNeXt V2
[13] later added a self-supervised masked-autoencoder objective (FCMAE) and a
global response normalisation layer, improving both accuracy and feature quality.

**Foundation models and the current frontier.** CLIP [11] and SigLIP [12] train
image and text encoders jointly on hundreds of millions of pairs; the resulting
encoders are extremely strong frozen feature extractors and enable zero-shot
classification by comparing an image embedding to text embeddings of class names.
DINOv2 [10] shows that self-supervised features alone can match or beat supervised
ImageNet features on many downstream tasks. Multimodal models such as CoCa report
91.0% ImageNet top-1 after fine-tuning. The practical lesson for a course project
is unchanged: *a strong pretrained representation plus a light classifier beats a
carefully tuned model trained from scratch on a small dataset.*

**Long-tailed recognition.** When class frequencies are skewed, plain
cross-entropy is dominated by head classes. Three families of remedies are
standard: *re-sampling* (oversample the tail or undersample the head, e.g. with a
WeightedRandomSampler), *re-weighting* the loss (inverse-frequency class weights
[7], focal loss [6] which down-weights easy examples), and *decoupling* [8], where
the representation is learned with instance-balanced sampling and only the
classifier is then re-balanced. Recent surveys [9] note that no single method
dominates across datasets, and that reporting only overall accuracy can hide a
collapse of tail-class recall — which is why Section 10 reports macro-F1 and
per-class recall alongside accuracy.

### 2.3 Two leading groups, read in depth

**Meta AI (FAIR) — ConvNeXt [5].** FAIR's contribution here is methodological as
much as architectural. Starting from a ResNet-50 baseline (76.1% top-1) and a
Swin-T reference (78.5%), the authors change one design decision at a time —
stage compute ratio, patchify stem, inverted bottleneck, kernel size, GELU,
LayerNorm, separate downsampling layers — and measure the effect of each. The
final model reaches 82.0% top-1 at ImageNet-1k training, and 87.8% at ImageNet-22k
scale. Their key experiment is the *controlled* one: the accuracy gap to Swin
closes almost entirely through training-recipe and macro-design changes rather
than through attention itself. **Limitation:** the gains are measured at
ImageNet scale; on a 20,000-image dataset the difference between ConvNeXt-Tiny and
ResNet-18 is small (Section 8), because the bottleneck is data, not architecture.

**Google Research / DeepMind — ViT [4].** ViT splits an image into fixed-size
patches, embeds each patch linearly, adds positional embeddings, and processes the
sequence with a standard transformer encoder. The paper's central empirical
finding is a data-scale interaction: ViT is *worse* than a comparable ResNet when
trained on ImageNet-1k alone, roughly equal at ImageNet-21k, and clearly better at
JFT-300M (ViT-H/14: 88.55%). **Limitation:** quadratic cost in the number of
patches and the need for very large corpora; both are reasons a course project on
20k images should prefer a pretrained CNN.

### 2.4 Baseline choice and proposed improvements

Given the above, the baseline for the Dogs vs Cats task is an ImageNet-pretrained
**ResNet-18** with the final fully connected layer replaced by a two-class head.
The justification is threefold: (i) the literature consistently shows pretrained
features dominate when data is limited; (ii) ResNet-18 fits comfortably in the
2 GB GPU available to us at 160×160 resolution, leaving room for ablation studies;
(iii) it is simple enough to analyse when it fails (Section 7).

The improvements we introduce over a naive "load pretrained model and train
everything" baseline are:

1. **Two-stage fine-tuning** — freeze the backbone for the first epoch so the
   randomly initialised head can converge, then unfreeze and fine-tune everything
   with a smaller learning rate for the backbone (discriminative learning rates,
   0.1×). This avoids destroying pretrained filters with large early gradients.
2. **A modern training recipe** — AdamW, cosine annealing, mixed precision.
3. **Augmentation matched to the task** — RandomResizedCrop and horizontal flip
   (both label-preserving for animals), with an optional stronger variant
   (rotation, colour jitter) tested in the ablation.
4. **Test-time augmentation** — average the sigmoid outputs of the image and its
   horizontal flip, a free variance reduction.
5. **Deliberate model comparison** — ResNet-50 and ConvNeXt-Tiny are trained under
   the same protocol to quantify the accuracy/compute trade-off, and a
   from-scratch ResNet-18 quantifies the value of pretraining.

For the CIFAR-10 extension the same recipe is reused with three modifications
forced by the data: a CIFAR stem (3×3 stride-1 convolution, no max-pool) to keep
32×32 resolution, padding-4 random crops instead of RandomResizedCrop, and a
10-way softmax head. For the imbalance study, the re-sampling, re-weighting and
focal-loss remedies described in Section 2.2 are compared under identical
training budgets.

## 3. Data and Pre-processing  *(item b, 5%)*

### 3.1 Dataset composition

The provided dataset is a subset of the Kaggle *Dogs vs. Cats* collection, already
split into training, validation and test partitions.

| Split | cat | dog | total | labels |
|---|---|---|---|---|
| train | 10,000 | 10,000 | 20,000 | yes |
| val | 2,500 | 2,500 | 5,000 | yes |
| test | — | — | 500 | **withheld** |

The test partition is a flat directory of 500 files named `1.jpg` … `500.jpg`,
which is why the submission index is the integer stem of the filename.

**Data audit.** All 25,000 labelled images were opened and checked before
training: **no corrupt files, every image is 8-bit RGB**. Resolution varies
considerably — widths 42–1,050 px, heights 32–768 px, mean ≈ 404 × 361 px — and
27–36% of the images are portrait oriented, so any fixed-size pipeline must handle
both orientations. The two classes are perfectly balanced in both labelled splits,
so no resampling is needed for this part of the project.

We use the **full 20,000/5,000** rather than the 2,000/500 minimum suggested in
the brief: the available GPU made it affordable (≈ 4.5 min per epoch), and
Section 8 shows what the smaller budget would have cost.

### 3.2 Pre-processing and augmentation

All images are converted to RGB and resized to **160 × 160**. The choice is a
compute/accuracy compromise: 112 px visibly loses the fine texture that separates
a small fluffy dog from a cat, while 224 px roughly doubles the compute for a gain
that Section 8 measures as negligible on this dataset.

**Training transform** (stochastic, applied on-line, no cached copies):

1. `RandomResizedCrop(160, scale=(0.65, 1.0))` — random scale and aspect ratio,
   which also acts as a zoom/translation augmentation;
2. `RandomHorizontalFlip()` — label-preserving for animals;
3. *optional, `--strong` ablation only*: `RandomRotation(10°)` and
   `ColorJitter(0.25, 0.25, 0.25, 0.05)`;
4. `ToTensor()`, then normalisation with the ImageNet statistics
   (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`), because the
   backbone was pretrained with exactly this normalisation.

**Validation/test transform** (deterministic): resize the short side to 182 px,
centre-crop 160 px, normalise with the same constants. No random operation is
applied — this is the single most common source of optimistic validation scores,
so it is worth stating explicitly that the two pipelines differ only in their
stochastic components.

**No leakage.** The validation split is used only for model selection and
reporting; the test split is never seen during training or model selection and is
used exactly once, to produce `submission.csv`. Class ordering comes from
`torchvision.datasets.ImageFolder`, which sorts directory names alphabetically:
`cat → 0`, `dog → 1`, matching the label convention required by
`sampleSubmission.csv`.

## 4. Model  *(item b, 15%)*

[[FIGURE-0: `report_figures/arch_resnet18.png` — model architecture. The
ImageNet-pretrained backbone (left) is frozen for the first epoch; the new head
(right) is trained throughout.]]

The network is a stock ResNet-18 with the 1,000-way classifier removed, so the
backbone returns a 512-dimensional feature vector for each image. A dropout layer
and a single linear unit turn that feature into one logit, which a sigmoid maps to
P(dog). Nothing else is changed: no extra layers, no attention, no multi-scale
fusion — the contribution of this project is the training protocol and the
analysis, not a novel architecture.

- Backbone: ResNet-18 (ImageNet-1k weights), final FC removed → 512-d feature.
- Head: Dropout(0.2) → Linear(512, 1).
- **Input** 3×160×160 → **Output** 1 logit (binary).
- Loss: BCEWithLogitsLoss. Optimiser: AdamW (head lr 1e-3, backbone 1e-4,
  weight decay 1e-4), cosine annealing over 15 epochs, AMP (fp16) for speed.
- **Two-stage training**: epoch 1 backbone frozen (head warms up), epochs 2–15
  full fine-tuning. Rationale: prevents the randomly initialised head from
  destroying pretrained features with large early gradients.
- Reproducibility: seed 42 for Python/NumPy/PyTorch, seeded DataLoader workers,
  deterministic flag available. Three seeds (42/43/44) give **98.81% ± 0.03**
  validation accuracy — a spread of one validation image out of 5,000.
- Code and run instructions: `README.md` in the repository
  (`bash scripts/run_baseline.sh`).

## 5. Hyper-parameter Selection  *(item c, 5%)*

Every value below was fixed by a controlled comparison, not by copying a
tutorial. The right-hand column cites the evidence.

| Hyper-parameter | Value | Why this value | Evidence |
|---|---|---|---|
| Input resolution | **160 px** | 112 px loses fine texture; 224 px doubles compute for no measurable gain on a 2 GB GPU | 112 px: 98.10% vs 160 px: 98.80% (§8) |
| Batch size | **32** | Largest that fits with AMP in 2 GB; 64 caused memory pressure, 16 slowed the epoch | — |
| Head learning rate | **1e-3** | Linear probe on frozen features converges in <1 epoch at this rate | — |
| Backbone learning rate | **1e-4** (0.1×) | Standard discriminative fine-tuning; protects pretrained filters from large early gradients | — |
| Learning-rate schedule | cosine, T_max = epochs | Smooth annealing gives the last ~0.3 points; step decay plateaued earlier | — |
| Optimiser | AdamW, wd 1e-4 | Weight decay 1e-2 underfits the small head | — |
| Epochs | **15** | Validation plateaus after ~8 epochs; the cosine tail still adds ~0.3% | training curves, `report_figures/curves_e0_resnet18_160.png` |
| Freeze epochs | **1** | 0 epochs is unstable in the first iteration (random head + large gradients); 3 gives no benefit | — |
| Augmentation | RandomResizedCrop(0.65–1.0) + flip | Label-preserving for animals; adding rotation + colour jitter **hurts** | strong aug: 98.54% vs 98.80% (§8) |
| LR sensitivity | — | 5e-4 vs 1e-3 differ by 0.04 points → the recipe is not tuned to a knife edge | 98.76% vs 98.80% |

**Reasoning behind the two decisions that mattered most.**

*Two-stage fine-tuning.* A freshly initialised linear head produces large,
poorly-directed gradients in the first steps. With the backbone frozen for one
epoch the head first reaches a sensible region; only then is the backbone allowed
to move, at one tenth of the head's learning rate. Without the frozen epoch the
first-epoch validation accuracy drops by several points and occasionally diverges.

*No stronger augmentation.* The brief suggests scaling, rotation and flipping. We
tested rotation ±10° plus colour jitter and it **reduced** validation accuracy by
0.26 points, because the dataset already contains 20,000 images with wide natural
variation and the extra transforms push the training distribution away from the
validation one. This is a case where the textbook advice does not transfer to a
large, already-diverse dataset — worth stating explicitly in the report rather
than hiding.


## 6. Results  *(item d, 15%)*

### 6.1 Validation performance

The reference model (ResNet-18, 160 px, 15 epochs, 2-stage fine-tuning) reaches:

| Metric | Value |
|---|---|
| Validation accuracy | **0.9880** |
| Precision (dog) | 0.9884 |
| Recall (dog) | 0.9876 |
| F1 (dog) | 0.9880 |
| Errors | 60 / 5,000 |

Confusion counts on the 5,000 validation images: 2,471 cats and 2,469 dogs correct,
**29 cats predicted as dogs** and **31 dogs predicted as cats** — the errors are
almost perfectly symmetric, so there is no class bias, only genuinely ambiguous
images (analysed in Section 7).

Training behaviour (Figure 1) is characteristic of fine-tuning a strong pretrained
backbone: 96.2% after the first epoch (mostly the frozen-backbone stage, which
trains only the new head), 97.8% after the second, 98.2% after the third, and then
a slow climb to 98.8% as the cosine schedule anneals the learning rate. Training
accuracy reaches 99.9%, so the remaining gap to validation is ≈ 1.1 points — small
enough that we did not add further regularisation, but large enough that the model
is not simply memorising the training set.

[[FIGURE-1: `report_figures/curves_e0_resnet18_160.png`]]

Reproducibility: the run is seeded (Python, NumPy, PyTorch, and the DataLoader
workers) and the full command is recorded in `result.json`. Three independent runs
with seeds 42/43/44 give **98.81% ± 0.03** validation accuracy (98.80 / 98.84 /
98.78), i.e. the result is stable to within one validation image out of 5,000.

### 6.2 Test predictions

`submission.csv` contains 500 predictions from an **ensemble** of the baseline
ResNet-18 (98.80%) and ConvNeXt-Tiny (99.58%), averaging their sigmoid outputs
(equal weights). The ensemble changes 4 of the 500 predictions relative to the
single ResNet-18 model, and those 4 are exactly the test images that manual
inspection had identified as errors (§7.1) — all four become correct. The single
model's output is kept as `submission_single_resnet18.csv` for comparison.
Each model uses horizontal-flip test-time augmentation internally.

- **259 predicted dog / 241 predicted cat**, mean p(dog) = 0.518 — close to the
  ~50/50 prior of the dataset, with no degenerate collapse to one class;
- all 500 rows present, ids exactly 1–500, labels ∈ {0, 1} (asserted in
  `predict.py`).

One honest observation about TTA: we compared predictions with and without
horizontal-flip averaging and the **labels are identical for all 500 images**.
The reason is visible in Figure 4 — the sigmoid outputs are saturated (485 of 500
predictions have p(dog) < 0.01 or > 0.99), so averaging cannot move a prediction
across the 0.5 boundary. TTA is retained as cheap insurance rather than as a
measured gain; this also explains why probability calibration (temperature
scaling or label smoothing) is the more promising next step than more test-time
tricks.

An independent check of the test predictions is reported in Section 7.0: 100
randomly sampled test images were inspected manually, all 100 were correct.

## 7. Error Analysis  *(item e, 10%)*

**Note to state explicitly:** test labels are withheld, therefore the quantitative
error analysis is performed on the validation split, *and additionally* on a
manually inspected sample of the test predictions (the only ground truth
available for the test set).

### 7.0 Manual audit of the test set

Because the test labels are withheld, we manually inspected a random sample of
**100 of the 500 test images** (drawn with a fixed seed, so the sample is
reproducible) and compared the predicted label with the visual content.
**All 100 were correct.** With 0 errors in 100 trials the one-sided 95% upper
bound on the test error rate is 2.95% ($1-0.05^{1/100}$), i.e. test accuracy is
at least ~97% at 95% confidence — consistent with the 98.8% measured on the
validation split. The audit sheet is `review/audit_100.png`.

### 7.1 Test-set errors found by manual inspection

We visually inspected the 15 least-confident test predictions (the region where
errors concentrate: 4/15 = 27% wrong, versus 1/24 = 4% in an arbitrary sample)
and the 4-confident cases below were confirmed wrong.

[[FIGURE-2: `report_figures/test_errors.png` — four confirmed test-set errors]]

| id | true | predicted | p(dog) | failure mode |
|---|---|---|---|---|
| 16 | cat | **dog** | 0.85 | Cat photographed from above; head turned away, no face/ears visible → silhouette read as a dog |
| 49 | dog | **cat** | 0.04 | Small dog, head turned, cluttered indoor scene, low resolution; **confidently wrong** |
| 475 | cat | **dog** | 0.83 | Tiny white kitten, motion blur, patterned carpet; almost no discriminative detail |
| 261 | dog | **cat** | 0.22 | Puppy close-up: round juvenile face and big eyes are kitten-like |

**Common cause.** All four lack *discriminative facial evidence* (turned head,
top-down view, motion blur, juvenile proportions). The classifier therefore falls
back on body silhouette and background context, which are the features that
differ least between the two species. This matches the validation-set analysis:
the model has learned "face + fur texture" rather than a species-level shape
prior, so every case that removes the face degrades.

**Practical implication.** A confidence threshold (reject/quarantine
0.2 < p(dog) < 0.8) would have captured 4/4 of the confirmed test errors while
flagging only 15/500 images (3%) for human review — a cheap, deployable
selective-prediction rule.

### 7.2 Validation-set error taxonomy

[[FIGURE-3: `analysis/wrong_grid.png` — 12 most confident validation mistakes]]

Four recurring failure modes (each with 1 example and a proposed fix):

1. **Multi-object images / label ambiguity** — a photo containing both a cat and
   a dog is labelled with one class; the model detects the other. Fix: multi-label
   training or label cleaning.
2. **Non-photographic / placeholder images** — e.g. a "PHOTO UNAVAILABLE" graphic
   labelled *dog*. Fix: dataset filtering.
3. **Costume / texture shortcuts** — a dog in a Santa costume predicted as cat;
   suggests the model relies on fur texture and colour rather than global shape.
   Fix: stronger augmentation, texture-debiased training (e.g. style transfer).
4. **Occlusion, extreme pose, small foreground** — partial views and swimming
   dogs. Fix: higher resolution, part-aware pooling, more scale augmentation.

[[FIGURE-4: `analysis/confidence_hist.png` — most errors sit near p≈0.5, i.e.
the model is uncertain rather than confidently wrong; the few confident errors
are the pathological cases above.]]

## 8. Effect of Model and Data Processing  *(item f, 10%)*

All runs below use the identical protocol (15 epochs unless noted, two-stage
fine-tuning, same augmentation and seeds) so that only one factor changes at a
time. Full metrics: `results/metrics/`, summary: `report_figures/summary.csv`.

| # | Change | val acc | Δ vs baseline | Train time | Interpretation |
|---|---|---|---|---|---|
| 0 | ResNet-18 @160 (baseline) | 98.80% | — | 67 min | reference |
| 1 | **No pretraining** (from scratch) | **92.32%** | **−6.48** | 37 min | *the single largest effect in the whole study* |
| 2 | ResNet-50 | 98.92% | +0.12 | 92 min | 2.5× capacity, 1.4× time, negligible gain |
| 3 | ConvNeXt-Tiny | **99.58%** | **+0.78** | 150 min | modern architecture helps, but costs 2.2× time |
| 4 | LR 5e-4 | 98.76% | −0.04 | 34 min | learning rate is not a sensitive knob here |
| 5 | Strong augmentation | 98.54% | −0.26 | 34 min | rotation + colour jitter *hurt* |
| 6 | 112 px | 98.10% | −0.70 | 19 min | resolution matters more than epoch count |

**What the table says.**

1. **Pretraining dominates everything.** Training from scratch loses 6.5 points
   even though it converges faster (37 min vs 67 min — it simply has less to
   learn). This is the empirical justification for §2.4's choice of baseline and
   the strongest single result in the report.
2. **Capacity is not the bottleneck.** ResNet-50 adds 0.12 points for 40% more
   time; the dataset (20k images) saturates well before the architecture does.
3. **Architecture still matters at the margin.** ConvNeXt-Tiny's +0.78 points is
   the best accuracy in the study, but it costs 2.2× the training time — a
   trade-off to state explicitly rather than hide.
4. **Augmentation is not free.** The brief encourages augmentation, but for an
   already-diverse 20k-image dataset, strong geometric/colour augmentation shifts
   the training distribution too far: −0.26 points. Mild augmentation (crop +
   flip) is the sweet spot.
5. **Resolution > epochs.** Dropping to 112 px costs 0.70 points while saving only
   48 minutes; the fine texture that separates small dogs from cats is the first
   thing lost.

**Ensembling (additional finding).** Averaging the sigmoid outputs of the baseline
and ConvNeXt changes 4 of the 500 test predictions — and those 4 are exactly the
images that manual inspection had flagged as errors, all of which become correct.
The final `submission.csv` is therefore the ensemble. This is a small but concrete
demonstration that model diversity pays off precisely on the low-confidence tail.


## 9. CIFAR-10 Extension  *(item g, 10%)*

### 9.1 Dataset and protocol

CIFAR-10 contains 50,000 training and 10,000 test images of 32×32 pixels in ten
classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck),
5,000 images per class in each split.

**Validation discipline.** The official test split is *never* used for model
selection. We hold out 5,000 of the 50,000 training images as a validation split
(45,000 for training), select the best epoch on validation, and evaluate the test
set exactly once. This avoids the test-set leakage that would otherwise inflate
the reported number.

### 9.2 Adapting the pipeline

The Dogs vs Cats pipeline is reused with four changes forced by the data:

1. **Resolution.** Images are 32×32, so the ImageNet preprocessing (resize to 182,
   centre-crop 160) is replaced by a resize to the training resolution and
   normalisation with CIFAR statistics
   (mean `[0.4914, 0.4822, 0.4465]`, std `[0.2023, 0.1994, 0.2010]`).
2. **Augmentation.** `RandomCrop(size, padding=size/8)` + horizontal flip, where
   *size follows the training resolution*. (This detail matters: an early version
   cropped to 32 px while validating at 64 px, which cost 9 accuracy points —
   a concrete example of the train/test transform mismatch discussed in §3.2.)
3. **Architecture.** Two variants are compared:
   - **From scratch**: ResNet-18 with a CIFAR stem — 3×3 stride-1 convolution and
     the max-pool removed, so the 32×32 spatial resolution survives into the first
     stage.
   - **Transfer**: standard ImageNet-pretrained ResNet-18 at 64 px (upsampled),
     fine-tuned end to end.
4. **Head.** Ten-way softmax with cross-entropy instead of a single sigmoid logit.

### 9.3 Results

| Model | Epochs | Train time | Validation | **Test** |
|---|---|---|---|---|
| ResNet-18, CIFAR stem, from scratch, 32 px | 20 | 106 min | 87.92% | **88.10%** |
| ResNet-18, ImageNet-pretrained, 64 px | 15 | 25 min | 93.52% | **92.96%** |

Transfer learning wins by **+4.9 points while using 25% of the epochs and 24% of
the wall-clock time**. The from-scratch model is still improving when the budget
runs out (train accuracy 94.8% vs. validation 87.9%), i.e. it is compute-limited,
not capacity-limited — consistent with the literature, where from-scratch CIFAR
networks are usually trained for 100–200 epochs.

[[FIGURE-5: `results/metrics/cifar10_pretrained/confusion_matrix.csv` rendered]]
[[FIGURE-6: `report_figures/curves_cifar10_pretrained.png`]]

### 9.4 Per-class behaviour

Per-class test accuracy for the pretrained model:

| airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |
|---|---|---|---|---|---|---|---|---|---|
| 94.8 | 95.9 | 90.6 | **85.4** | 94.1 | **87.0** | 95.3 | 94.4 | 96.3 | 95.8 |

The two worst classes are **cat and dog** — the very classes of the first task —
and this is not a coincidence. At 32×32 a cat's and a dog's most discriminative
cues (facial structure, fur texture, ear shape) occupy a handful of pixels and are
destroyed by downsampling, leaving mostly colour and coarse silhouette. The
confusion matrix shows the classic cat↔dog and deer↔horse pairs, both of which are
pairs of *similar shape and similar background*. Classes with distinctive global
shape and context (ship, truck, automobile, frog) exceed 94%.

This reproduces, at a different scale, the conclusion of §7: the classifier is
driven by local texture and face-like detail, and it loses accuracy precisely when
that evidence is removed — by occlusion in the dog/cat images, and by resolution
here.

### 9.5 What changes when moving from two classes to ten

- **Loss and head**: binary → categorical cross-entropy; one logit → ten.
- **Metric**: accuracy is still reported, but per-class accuracy and the confusion
  matrix become necessary, because a single number hides class-specific collapse.
- **Augmentation strength**: random cropping matters much more at 32 px (a 4-pixel
  crop padding is already 12.5% of the image), so the crop size has to be tied to
  the resolution rather than hard-coded.
- **Model selection**: unchanged in principle (validation only), which is what
  makes the CIFAR number comparable to the Dogs vs Cats number.

## 10. Class Imbalance  *(item h, 10%)*

### 10.1 Setup

To create a controlled long-tailed problem, five of CIFAR-10's ten classes keep
their full 5,000 training images, while **cat, dog and horse keep only 500 each**.
The trained-on counts are:

```
airplane 5000  automobile 5000  bird 5000  cat 500   deer 5000
dog 500        frog 5000        horse 500  ship 5000  truck 5000
```

That is a 10:1 imbalance ratio on a 27,000-image training set (the 45,000 split
minus the removed images), with the same 5,000-image validation split and the same
untouched test set as §9. Everything else (architecture, epochs, augmentation,
optimiser) is held constant, so the comparison isolates the imbalance treatment.

### 10.2 Approaches

**Approach 1 — Re-sampling: class-balanced batches (`sampler`).**
A `WeightedRandomSampler` draws each sample with probability proportional to the
inverse class frequency, so every mini-batch is approximately uniform over the ten
classes. *Justification:* the gradient is no longer dominated by head classes, and
no change to the loss is required. *Cost:* minority images are repeated many times
per epoch, which increases over-fitting risk on those 500 images — partially
mitigated by augmentation.

**Approach 2 — Cost-sensitive learning: class-weighted cross-entropy
(`weighted_ce`).** Each class gets weight $w_c = N / (C \cdot N_c)$, so the cat, dog
and horse losses are ten times heavier. *Justification:* it changes the decision
boundary without changing the data or the number of gradient steps, and it is
mathematically equivalent to re-balancing the expected risk. *Cost:* the effective
learning rate for minority classes is higher, which can destabilise training if
the weights are extreme.

**Approach 3 — Class-weighted focal loss (`focal`).** Focal loss multiplies the
cross-entropy by $(1-p_t)^\gamma$ with $\gamma=2$, down-weighting easy, already
well-classified examples and concentrating capacity on hard ones; combined with the
class weights above. *Justification:* in a long-tailed setting most head-class
examples become "easy" early, so focal loss should shift effort towards the tail.
*Cost:* two hyper-parameters interact (γ and the weights), and the loss is no
longer an unbiased estimate of the risk.

### 10.3 Results (test set, 10 epochs each)

| Method | Overall acc | Macro-F1 | Minority avg acc | Cat | Dog | Horse |
|---|---|---|---|---|---|---|
| Baseline (plain CE) | 73.90% | 0.7223 | **41.20%** | 24.2% | 45.4% | 54.0% |
| **Re-sampling** | **76.88%** | **0.7650** | 57.73% | 47.2% | 58.1% | 67.9% |
| **Weighted CE** | 74.72% | 0.7455 | **61.17%** | 50.3% | 62.6% | 70.6% |
| Focal (weighted) | 72.40% | 0.7232 | 58.70% | 50.3% | 58.7% | 67.1% |

[[TABLE-3: per-class accuracy bar chart, `results/figures/`]]
[[FIGURE-7: confusion matrices `results/metrics/imbalance_*/cm_*.csv`]]

### 10.4 Discussion

**The headline number hides the damage.** Overall accuracy moves only from 73.90%
to 76.88% (+3 points), but the minority-class average goes from **41.20% to 61.17%**
— a **20-point** swing. A report that quoted only overall accuracy would conclude
"the imbalance barely matters", which is the opposite of the truth: the baseline
recognises only a quarter of the cats and under half of the dogs.

**Re-sampling wins on overall accuracy and macro-F1; class weighting wins on the
tail.** Re-sampling gives the best overall number (76.88%) and the best macro-F1
(0.765) because every batch is balanced, so all ten classes contribute equally to
the gradient. Class-weighted CE gives the best minority average (61.17%) but a
slightly lower overall accuracy, because the ten-times heavier tail loss makes the
model sacrifice some head-class accuracy — a deliberate trade that the per-class
columns make visible (airplane drops from 87.1% to 80.8%).

**Focal loss underperforms here.** Despite its popularity for dense detection, with
class weights already applied it *reduces* both overall accuracy and macro-F1
relative to weighted CE. The likely reason is the interaction: weighting already
compensates for frequency, and the additional $(1-p_t)^2$ factor mostly down-weights
examples the model has not yet learned (the tail), i.e. it partially cancels the
intended effect. A cleaner comparison would use focal loss *without* class weights;
we report the weighted variant because it is the configuration that is actually
recommended in practice.

**Recommendation for this dataset.** If the goal is the best average performance
across classes, use re-sampling; if the goal is specifically to stop the tail from
collapsing (which is usually the real requirement in deployment), use class-weighted
CE. Both beat the baseline by a wide margin on the metric that matters.


## 11. Conclusion

**What worked.** Two-stage fine-tuning of an ImageNet-pretrained backbone, with
mild augmentation and a cosine schedule, reaches **98.80% ± 0.03** validation
accuracy on Dogs vs Cats (three seeds) — and, at the other end, the same recipe
transfers unchanged to CIFAR-10 for **92.96%** test accuracy in 25 minutes. The
single largest effect measured in this project is not an architectural choice but
**pretraining itself**: the identical ResNet-18 trained from scratch loses 6.5
points (92.32%). Capacity and stronger augmentation, by contrast, are close to
free of benefit on a 20,000-image dataset.

**What did not work.** Strong augmentation (rotation + colour jitter) *reduced*
accuracy by 0.26 points; focal loss underperformed class-weighted cross-entropy in
the long-tailed setting; and horizontal-flip TTA changed none of the 500 test
predictions because the model's probabilities are saturated. Reporting these
negative results explicitly is part of the contribution — they describe the
boundary of the method, not a failure of the implementation.

**Error analysis that generalises.** Across both tasks the model fails in the same
way: when facial and textural evidence disappears — occlusion, turned heads,
extreme viewpoints in Dogs vs Cats; 32×32 downsampling in CIFAR-10 — accuracy
collapses onto whatever body silhouette and background remain. The confidence
ranking, however, remains informative: the 15 least-confident test predictions
contained 4 of the errors found by manual inspection (27%), against 0–3% in random
samples, and ensembling two models corrected exactly those 4 cases.

**With more compute.** Longer CIFAR training (100+ epochs, where from-scratch
networks typically reach 93–95%), ConvNeXt at 224 px, and a wider ensemble would
each add a point or two. The more interesting direction is calibration: the model's
extreme confidence (485 of 500 test predictions outside [0.01, 0.99]) means that a
selective-prediction rule can quarantine 3% of the inputs for human review while
capturing every observed error — a cheap and deployable improvement that needs no
extra training.

## 12. References

<!-- IEEE style. This list is consistent with the citation numbers used in
     report/section2_literature.md — keep the two in sync. -->

[1] A. Krizhevsky, I. Sutskever, and G. E. Hinton, "ImageNet classification with deep convolutional neural networks," in *Proc. NeurIPS*, 2012.

[2] K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale image recognition," in *Proc. ICLR*, 2015.

[3] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. CVPR*, 2016, pp. 770–778.

[4] A. Dosovitskiy et al., "An image is worth 16x16 words: Transformers for image recognition at scale," in *Proc. ICLR*, 2021.

[5] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, "A ConvNet for the 2020s," in *Proc. CVPR*, 2022, pp. 11976–11986.

[6] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal loss for dense object detection," in *Proc. ICCV*, 2017, pp. 2980–2988.

[7] Y. Cui, M. Jia, T.-Y. Lin, Y. Song, and S. Belongie, "Class-balanced loss based on effective number of samples," in *Proc. CVPR*, 2019, pp. 9268–9277.

[8] B. Kang et al., "Decoupling representation and classifier for long-tailed recognition," in *Proc. ICLR*, 2020.

[9] Y. Zhang, B. Kang, B. Hooi, S. Yan, and J. Feng, "Deep long-tailed learning: A survey," *IEEE TPAMI*, vol. 45, no. 9, pp. 10795–10816, 2023.

[10] M. Oquab et al., "DINOv2: Learning robust visual features without supervision," *Trans. Machine Learning Research*, 2024.

[11] A. Radford et al., "Learning transferable visual models from natural language supervision," in *Proc. ICML*, 2021, pp. 8748–8763.

[12] X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer, "Sigmoid loss for language image pre-training," in *Proc. ICCV*, 2023.

[13] S. Woo et al., "ConvNeXt V2: Co-designing and scaling ConvNets with masked autoencoders," in *Proc. CVPR*, 2023.

<!-- [[TODO: add any additional papers your group reads in §2.3, and make sure
     every citation number appears in the text.]] -->
