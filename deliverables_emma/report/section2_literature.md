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
