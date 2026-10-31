# 1. Literature Survey

> EE6483 Mini Project (Option 2: Dogs vs. Cats)
> Group members and contributions: see front matter.

## 1.1 Problem definition, settings and challenges

### 1.1.1 Formal definition

Image classification is the task of learning a mapping $f_\theta : \mathcal{X} \to \mathcal{Y}$ from an image $x \in \mathbb{R}^{H \times W \times 3}$ to a label $y \in \{1,\dots,C\}$, by minimising the empirical risk

$$\hat{\theta} = \arg\min_\theta \frac{1}{N}\sum_{i=1}^{N} \ell\big(f_\theta(x_i), y_i\big)$$

over a training set $\{(x_i,y_i)\}_{i=1}^N$ drawn from a data distribution $P_{train}$. The project instance is the special case $C=2$ with $\mathcal{Y}=\{\text{cat},\text{dog}\}$: a supervised, closed-set, single-label, in-distribution binary classification problem. The evaluation protocol is accuracy on a held-out set drawn from the same distribution, plus a 500-image unlabelled test set whose predictions are submitted as `submission.csv`.

### 1.1.2 Why this "easy" problem is not easy

Humans solve cat/dog discrimination at above 99.9% accuracy, yet the problem concentrates almost every difficulty that makes visual recognition hard:

1. Large intra-class variation. "Dog" covers 340+ breeds that differ in size (Chihuahua vs. Great Dane, a 40× range in body mass), ear shape, coat length and colour. The same pixel-level variation exists within "cat". A classifier must be invariant to all of it while remaining sensitive to the much smaller cat/dog differences.
2. Small inter-class distance in specific sub-populations. Puppies and kittens share body proportions (large head-to-body ratio, short snout); long-haired cats and small fluffy dogs (Pomeranian, Shih Tzu) share texture. These are the samples that dominate the residual error of any strong model.
3. Background shortcut learning. In the original Kaggle distribution, dogs are photographed outdoors on grass and cats indoors on sofas. A model can reach surprisingly high accuracy by recognising *context* rather than *animal*. This is a spurious correlation in the sense of shortcut learning; it inflates validation accuracy and collapses under distribution shift. Random erasing / aggressive cropping are the standard mitigations, and Grad-CAM is the standard diagnostic (Section 6).
4. Pose, occlusion, scale and illumination. Animals are articulated and non-rigid; they appear at arbitrary scale, partially occluded, in arbitrary lighting. Global average pooling over a deep feature map handles translation and scale reasonably, but not occlusion.
5. Label noise. Web-scraped datasets contain mislabelled images, cartoon drawings, and images with both a cat and a dog. With ~2% noise the achievable accuracy is capped below 100% no matter how good the model is; this is visible in our error analysis as "confident errors on ambiguous images".
6. Small-data regime. 20,000 training images is small for training a deep network from scratch (ImageNet has 1.28M), which motivates transfer learning.

### 1.1.3 Settings: supervised vs. unsupervised, closed-set vs. open-set, with vs. without domain shift

The literature organises classification methods along three orthogonal axes. Stating where this project sits on each axis is what makes the design choices defensible.

Supervision level.

| Setting | Labelled data | Representative methods | Relevance here |
|---|---|---|---|
| Supervised | all training data labelled | AlexNet, VGG, ResNet, ViT | **this project** |
| Semi-supervised | small labelled + large unlabelled | pseudo-labelling, MixMatch, FixMatch [Sohn et al., NeurIPS 2020] | would exploit the 500 test images, which is not permitted |
| Self-supervised pre-training | none (pretext task), then linear probe / fine-tune | MoCo, SimCLR, BYOL, MAE, DINOv2 | source of the pretrained weights we rely on |
| Few-shot / metric learning | a handful per class | Prototypical networks, MAML | not applicable (10k per class) |
| Zero-shot | none, class described in text | CLIP-style vision-language models | interesting upper bound; not needed |

Closed-set vs. open-set. In a *closed-set* problem every test image is assumed to belong to one of the known classes. In an *open-set* problem the test set contains images of unknown classes and the system must be able to say "none of the above". The consequence for this project is concrete: if the test set contained an image of a rabbit, our softmax classifier is *forced* to output either "cat" or "dog", and it will do so with high confidence. Standard remedies are thresholding the maximum softmax probability [Hendrycks & Gimpel, ICLR 2017], OpenMax [Bendale & Boult, CVPR 2016], and energy-based scores; all of them are post-hoc and could be bolted onto our model without retraining. We explicitly discuss this limitation in Section 6.

With vs. without domain shift. A *domain shift* means $P_{test} \neq P_{train}$. The two dominant sub-problems are domain adaptation (unlabelled target data available) [Wang & Deng, Neurocomputing 2018] and domain generalisation (only source data available). The robustness literature [Hendrycks & Dietterich, ICLR 2019] shows that ImageNet-trained models lose 20–40 accuracy points under mild corruptions (blur, noise, JPEG artefacts, contrast change) even though clean accuracy is unchanged. Section 6 therefore includes a corruption test, since two models with identical clean accuracy can differ by ten points under blur.

### 1.1.4 Taxonomy of solution families

1. Hand-crafted features + shallow classifier (1990s–2012). Colour histograms, SIFT/HOG descriptors, then SVM or random forest. On cats vs. dogs these saturate around 65–75%. They remain a useful *lower bound* and a reminder of how much of the gain comes from representation learning [LeCun et al., Nature 2015]; we implement one (M0) for exactly this reason.
2. CNNs trained end-to-end (2012–2019). AlexNet [Krizhevsky et al., NeurIPS 2012] established the recipe (ReLU, dropout, GPU, augmentation) that is still in use. Depth then increased through VGG [Simonyan & Zisserman, ICLR 2015], GoogLeNet [Szegedy et al., CVPR 2015], ResNet [He et al., CVPR 2016], DenseNet [Huang et al., CVPR 2017] and ResNeXt [Xie et al., CVPR 2017].
3. Efficient architecture search and scaling (2019–2021). EfficientNet [Tan & Le, ICML 2019] showed that depth/width/resolution must be scaled jointly under a compound coefficient.
4. Transformers and large-scale pre-training (2020–). ViT [Dosovitskiy et al., ICLR 2021] replaced convolution with global self-attention and demonstrated that *data scale*, not inductive bias, is the binding constraint; Swin [Liu et al., ICCV 2021] reintroduced locality through windowed attention; ConvNeXt [Liu et al., CVPR 2022] showed a modernised ResNet matches Swin, i.e. the architecture family matters less than the training recipe.
5. Transfer learning (the practical workhorse). Fine-tune an ImageNet-pretrained backbone; with 20k images this beats training from scratch by 10+ points. The two key papers are Yosinski et al. [NeurIPS 2014] on feature transferability and Kornblith et al. [CVPR 2019] on the correlation between ImageNet accuracy and transfer accuracy.
6. Long-tailed / imbalanced learning. Re-weighting [Cui et al., CVPR 2019; Cao et al., NeurIPS 2019], re-sampling, decoupled training [Kang et al., ICLR 2020], post-hoc logit adjustment [Menon et al., ICLR 2021]. Covered in Section 9.

## 1.2 Recent progress: papers that shaped the field

The table below is the reading list behind this report; the "role here" column states why each entry matters for the project rather than merely summarising it.

| Work | Venue | Key idea | Role here |
|---|---|---|---|
| Krizhevsky et al., *ImageNet Classification with Deep CNNs* | NeurIPS 2012 | ReLU + dropout + GPU + augmentation | origin of the modern training recipe we use |
| Simonyan & Zisserman, *Very Deep CNNs* | ICLR 2015 | stacked 3×3 kernels; depth 16–19 | the named baseline backbone |
| He et al., *Deep Residual Learning* | CVPR 2016 | identity shortcut solves degradation | **baseline architecture (M3–M5)** |
| He et al., *Identity Mappings in ResNet* | ECCV 2016 | pre-activation; cleaner gradient flow | motivates BN/ReLU placement |
| Huang et al., *DenseNet* | CVPR 2017 | dense connectivity, feature reuse | alternative to residual connections |
| Xie et al., *ResNeXt* | CVPR 2017 | grouped convolutions, cardinality | width-vs-depth trade-off |
| Tan & Le, *EfficientNet* | ICML 2019 | compound scaling | reference point for "bigger is better, if scaled right" |
| Dosovitskiy et al., *ViT* | ICLR 2021 | patch embedding + self-attention | shows data scale dominates inductive bias |
| Liu et al., *Swin Transformer* | ICCV 2021 | shifted-window attention | hierarchical transformer |
| Liu et al., *A ConvNet for the 2020s* | CVPR 2022 | modernised ResNet ≈ Swin | justifies staying with ResNet |
| Radford et al., *CLIP* | ICML 2021 | image–text contrastive pre-training | zero-shot transfer; open vocabulary |
| He et al., *MoCo* | CVPR 2020 | momentum queue for contrastive learning | self-supervised pre-training |
| He et al., *MAE* | CVPR 2022 | masked patch reconstruction | self-supervised pre-training |
| Oquab et al., *DINOv2* | TMLR 2024 | self-supervised features that transfer without labels | state of the art frozen features |
| Yosinski et al., *How transferable are features?* | NeurIPS 2014 | layer-wise transferability, co-adaptation | justifies freezing-then-unfreezing |
| Kornblith et al., *Do Better ImageNet Models Transfer Better?* | CVPR 2019 | ImageNet acc. predicts transfer acc. | justifies choosing a stronger backbone |
| Kumar et al., *Fine-Tuning can Distort Pretrained Features* | ICLR 2022 | LP-FT: linear probe then fine-tune | **our training strategy (M3)** |
| Cubuk et al., *RandAugment* | CVPRW 2020 | parameter-free augmentation policy search | augmentation component |
| Zhang et al., *mixup* | ICLR 2018 | linear interpolation of inputs/labels | optional regulariser |
| Yun et al., *CutMix* | ICCV 2019 | patch-level mixing with label proportion | optional regulariser |
| DeVries & Taylor, *Cutout* | arXiv 2017 | random erasing as regularisation | motivates RandomErasing |
| Cui et al., *Class-Balanced Loss* | CVPR 2019 | effective number of samples | Section 9 method 1 |
| Cao et al., *LDAM-DRW* | NeurIPS 2019 | class-dependent margin + deferred re-weighting | Section 9 method 2 |
| Kang et al., *Decoupling Representation and Classifier* | ICLR 2020 | representation is unbiased, classifier is not | Section 9 method 3 |
| Menon et al., *Logit Adjustment* | ICLR 2021 | Bayes-optimal post-hoc shift | Section 9 method 4 |
| Lin et al., *Focal Loss* | ICCV 2017 | down-weight easy examples | Section 9 ablation |
| Selvaraju et al., *Grad-CAM* | ICCV 2017 | gradient-weighted class activation maps | Section 6 diagnostic |
| Hendrycks & Dietterich, *ImageNet-C* | ICLR 2019 | corruption robustness benchmark | Section 6 robustness test |
| Bendale & Boult, *OpenMax* | CVPR 2016 | open-set rejection | limitation discussion |
| Elson et al., *Asirra* | CCS 2007 | the original cat/dog CAPTCHA dataset | historical baseline |
| Krizhevsky, *Learning Multiple Layers of Features from Tiny Images* | TR 2009 | the CIFAR datasets | Section 8/H dataset |

Two observations from this list are worth stating explicitly because they shape our design:

- Architecture progress has plateaued for this task size. ConvNeXt [Liu et al., CVPR 2022] shows that a ResNet trained with a modern recipe matches a Swin transformer; Kornblith et al. [CVPR 2019] show that ImageNet accuracy, rather than architectural novelty, predicts transfer accuracy. For a 20k-image binary problem the marginal return of switching from ResNet to ConvNeXt or ViT is expected to be small; our controlled sweep (Section 7.8) measures it at 0.36 pp for ConvNeXt-T over ResNet-18, and none for ViT-B/16 at the same budget. We therefore invest in the *training recipe* (augmentation, LP-FT, label smoothing, TTA, seed averaging) rather than in exotic backbones.
- The bottleneck moved from representation to data and evaluation. With ImageNet pre-training, cat/dog accuracy is essentially saturated (Section 1.4), so the informative experiments are the ones that *remove* the advantage: train from scratch, freeze the backbone, shrink the training set. This is why our ablation matrix (Section 7) is built around data quantity and transfer strategy rather than around architecture.

## 1.3 Two leading groups and a close reading of their work

### 1.3.1 Microsoft Research Asia (MSRA): the residual-learning line

The Visual Computing Group at MSRA, led by Kaiming He with Xiangyu Zhang, Shaoqing Ren and Jian Sun, produced the single most cited architecture in vision. Their contribution is best understood as a diagnosis followed by a fix.

Diagnosis (He et al., CVPR 2016). The paper starts from an empirical puzzle: a 56-layer plain network has *higher* training error than a 20-layer one. This is not over-fitting (training error is worse) and not vanishing gradients in the classical sense (BN already keeps gradients healthy). The authors name the phenomenon degradation: optimising a deeper stack of non-linear layers is harder than optimising a shallower one plus the identity, even though the deeper model can represent the shallower solution exactly.

Fix. Re-parameterise each block to learn a residual $F(x) = H(x) - x$, so the block output is $F(x) + x$. If the identity is optimal, driving $F$ to zero is easier than learning the identity through non-linearities. The paper also shows that shortcut connections add neither parameters nor computational cost, which is what makes depth "free".

Evidence and impact. the 152-layer ResNet won ILSVRC 2015 with 3.57% top-5 error, and the ablation is unusually clean: plain vs. residual at 18/34 layers, with and without shortcuts, isolates exactly one variable. The follow-up [He et al., ECCV 2016] moved BN and ReLU before the convolution (pre-activation) and improved optimisation further, which is why modern ResNets use `conv-BN-ReLU` ordering and a zero-initialised last BN in each block.

Why it matters for us. Our backbone is a ResNet, and the LP-FT finding [Kumar et al., ICLR 2022] that fine-tuning *distorts* pretrained features is essentially a statement about how much of the residual network should be allowed to move. A frozen ResNet-18 with a linear head (M2) is a strong feature extractor; unfreezing everything at a large learning rate destroys some of that structure before the head has learned to use it: which is exactly the accuracy dip we observe at the first unfrozen epoch and why we warm up the backbone learning rate.

### 1.3.2 Google Research: the scaling and pre-training line

The second line of work asks a different question: not "how do we build a better block?" but "what happens if we scale data, parameters and compute together, and how do we pre-train without labels?"

- EfficientNet (Tan & Le, ICML 2019). The central claim is that width, depth and input resolution must be scaled *jointly* with a compound coefficient $\phi$: $d=\alpha^\phi, w=\beta^\phi, r=\gamma^\phi$ subject to $\alpha\beta^2\gamma^2\approx 2$. Scaling only one axis saturates quickly. The paper combines this with a neural-architecture-search-derived baseline (MBConv + squeeze-and-excitation). For us the practical lesson is that resolution is a first-class hyper-parameter, which is why our ablation sweeps 96/128/160/224 px and why the CIFAR-10 experiment must change the stem rather than simply upscale.
- ViT (Dosovitskiy et al., ICLR 2021). A pure transformer over 16×16 patches, with almost no image-specific inductive bias. The key empirical finding is that ViT underperforms ResNet when trained on ImageNet alone but *overtakes* it when pre-trained on ImageNet-21k or JFT-300M. The conclusion that inductive bias can be replaced by data reframed the field and led directly to the current dominance of large-scale pre-training.
- ViT follow-ups. DeiT introduced strong augmentation and distillation to train ViT on ImageNet-1k alone; Swin reintroduced a hierarchical, locality-aware design; ConvNeXt closed the gap from the convolutional side. DINOv2 [Oquab et al., TMLR 2024] then showed that self-supervised ViT features are excellent *frozen* features, which is the modern counterpart of our M2 linear-probe experiment.

Synthesis of the two groups. MSRA answers "how do we make optimisation of deep networks work?"; Google answers "how do we exploit scale and unlabelled data?". Our project sits exactly at their intersection: we use a residual architecture (MSRA) pre-trained on ImageNet (the Google-scale paradigm) and adapt it to a small dataset with a careful fine-tuning schedule (the LP-FT insight).

## 1.4 Choice of baseline and the improvements we implement

### 1.4.1 Baseline

Baseline: ImageNet-1k pre-trained ResNet-18, fully fine-tuned at 224×224, cross-entropy with label smoothing, AdamW, cosine schedule.

Justification:

- ResNet is explicitly permitted by the assignment and is the most reproducible choice available in `torchvision`.
- ResNet-18 (11.2M parameters) trains in ~7 minutes per run on the available 8 GB laptop GPU, which makes a three-seed protocol and a 17-run ablation matrix affordable. ResNet-50 is kept as a scaling check, not as the main model.
- Kornblith et al. [CVPR 2019] show that within a family, higher ImageNet accuracy transfers better; across families, the differences are small relative to the recipe. Given a fixed compute budget, spending it on seeds and ablations is more informative than spending it on a bigger backbone.

### 1.4.2 ImageNet already contains cats and dogs

ImageNet-1k [Russakovsky et al., IJCV 2015] includes roughly 118 dog breeds (Norwegian elkhound, Greater Swiss mountain dog, …) and several cat species (tabby, tiger cat, Persian, Siamese, Egyptian cat). A pre-trained backbone has therefore already solved a *fine-grained* version of this task, and the 2-way problem is close to a subset of its training objective. This means:

1. accuracy on this dataset is near-saturated (99%+), so headline accuracy is a weak discriminator between methods;
2. part of the measured "transfer" is genuine category prior rather than generic visual representation;
3. any claim that "our model learned to distinguish cats from dogs" must be supported by evidence that does not depend on pre-training.

We therefore build the report around three controls that remove the prior: (i) a SmallCNN trained from scratch, (ii) a frozen-backbone linear probe, and (iii) a learning curve over 500 → 20,000 training images. If the pre-trained model's advantage disappears when the data is plentiful and reappears when data is scarce, that is evidence of *transfer*, not of leakage.

### 1.4.3 Improvements over the baseline, each mapped to an experiment

| # | Improvement | Motivation | Verified in |
|---|---|---|---|
| I1 | LP-FT: 2 epochs linear probe, then full fine-tune with 3× lower backbone LR | Kumar et al. [ICLR 2022]: fine-tuning distorts features, especially under distribution shift | Section 7 (ft-strategy) |
| I2 | Strong augmentation (RRC + flip + rotation + ColorJitter + RandAugment + RandomErasing) | small dataset; RandomErasing also attacks the background shortcut | Section 7 (aug) |
| I3 | Label smoothing ε=0.1 | web labels are noisy; prevents over-confident logits and improves calibration | Section 4 |
| I4 | Cosine schedule with warm-up, gradient clipping, AMP | stable optimisation at batch 64 on 8 GB | Section 4 |
| I5 | Horizontal-flip TTA + 3-seed softmax averaging | variance reduction; cheap and reproducible | Section 5 |
| I6 | Resolution sweep 96→224 | EfficientNet's lesson that resolution is a first-class axis | Section 7 (resolution) |
| I7 | Stronger backbone (ResNet-34/50) | Kornblith et al. [CVPR 2019] | Section 7 (model) |

The survey therefore produces the baseline and the list of candidate improvements; Sections 3 to 5 implement them and Section 7 removes them one at a time to measure what each contributes.
