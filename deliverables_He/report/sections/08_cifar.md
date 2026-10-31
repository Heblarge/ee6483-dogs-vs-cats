# 8. Extension to CIFAR-10

## 8.1 The dataset and the classification problem

CIFAR-10 [Krizhevsky, TR 2009] consists of 60,000 colour images of 32×32 pixels in 10 mutually exclusive classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. The official split is 50,000 training and 10,000 test images, 6,000 per class overall, so the class distribution is exactly balanced and the majority-class baseline is 10%.

Compared with Dogs vs. Cats, four properties change the problem:

| Property | Dogs vs. Cats | CIFAR-10 | Consequence |
|---|---|---|---|
| Classes | 2 | 10 | output layer 2 → 10; chance level 50% → 10% |
| Resolution | ~400×400 native, 224 used | 32×32 | the object occupies ~15×15 pixels; fine texture is gone, colour and silhouette dominate |
| Object scale | large, centred | small, often off-centre | global pooling still works, but the stem must not over-downsample |
| Intra-class structure | breeds within one species | 10 unrelated categories | errors concentrate between *semantically* similar pairs (cat↔dog, deer↔horse, automobile↔truck) |
| Training size | 20,000 | 50,000 | from-scratch training becomes viable |

The last point matters: on 20,000 images a from-scratch CNN underfits badly, whereas 50,000 images of 32×32 are enough for a small ResNet to reach the low 90s. CIFAR-10 is therefore the right test of whether our pipeline generalises, and it exposes exactly the design decisions that were previously hidden by pre-training.

The 10,000-image official test set is used once, at the end of each run, on the checkpoint that validation accuracy selected. It plays no part in model selection, early stopping or hyper-parameter choice, so the test numbers reported in Section 8.4 are not the result of tuning against them.

## 8.2 What has to change in the code

| Component | Dogs vs. Cats | CIFAR-10 | Why |
|---|---|---|---|
| Output head | `Linear(512→2)` | `Linear(512→10)` | 10 classes |
| **Stem** | 7×7 conv, stride 2, then 3×3 max-pool, stride 2 | **3×3 conv, stride 1, no max-pool** | see below |
| Input | 224×224 (Resize 256 → CenterCrop) | 32×32 native, or 32→160 upsampled for the pre-trained model | native resolution for scratch training |
| Augmentation | RRC(0.5–1.0) + flip + rotation + jitter + RandAugment + erasing | RandomCrop(32, pad=4) + flip + RandAugment(m=9) + erasing | no room for scale jitter at 32 px; padding-crop is the CIFAR standard |
| Normalisation | ImageNet mean/std | ImageNet mean/std (same) | shared code path, no measurable difference |
| Validation split | provided (5,000) | **carved out of the 50,000: 45,000 / 5,000, stratified** | CIFAR-10 has no official validation set; using the test set for early stopping would leak |
| Optimiser | AdamW, lr 3e-4 | SGD + momentum 0.9 + Nesterov, lr 0.1, cosine, wd 5e-4 | the standard CIFAR recipe; AdamW is slower to converge here |
| Schedule | 8 epochs | 25 epochs (scratch) / 7 epochs (fine-tune) | small images need many more passes |
| Metrics | accuracy, macro-F1, AUC | accuracy, macro-F1, per-class recall, 10×10 confusion matrix | per-class behaviour is the interesting part |

### Why the stem is the critical change

The ImageNet stem applies a 7×7 convolution with stride 2 followed by a 3×3 max-pool with stride 2, i.e. an overall downsampling factor of 4 before the first residual stage. On a 224×224 ImageNet image this leaves 56×56: sensible. On a 32×32 CIFAR image it leaves 8×8, at which point each spatial location covers a 4×4 pixel patch of the original image; the first residual stage then operates on an almost-degenerate feature map, and after `layer4` the representation collapses to 1×1. The network is effectively forced to make its decisions from a handful of coarse cells.

Replacing the stem with a single 3×3 stride-1 convolution and removing the max-pool keeps the feature map at 32×32 after `conv1`, so the four stages produce 32 → 32 → 16 → 8 → 4. This is the standard "CIFAR ResNet" modification and it is worth roughly 3–5 accuracy points at this dataset size: a much larger effect than any hyper-parameter in Section 4.

`ee6483/src/models/backbones.py::build_cifar_resnet` implements exactly this and is used by G1 and by all Section 9 experiments.

## 8.3 Two models

G1: CIFAR-ResNet-18 trained from scratch. 11.2M parameters, 3×3 stride-1 stem, SGD (lr 0.1, momentum 0.9, Nesterov, wd 5e-4), cosine schedule with 5% warm-up, batch 128, 25 epochs, standard CIFAR augmentation. No pre-training of any kind.

G2: ImageNet ResNet-18, upsampled inputs. The 32×32 images are resized to 160×160 with bicubic interpolation before entering the standard ImageNet stem, the head is replaced by `Linear(512→10)`, and the network is fine-tuned for 7 epochs with SGD (lr 0.01, cosine). This isolates the value of pre-training from the value of architecture.

## 8.4 Results

| Model | Init | Epochs | Val acc | **Test acc** | Test macro-F1 | Train time |
|---|---|---|---|---|---|---|
| G1 CIFAR-ResNet-18 | random | 25 | 93.44% | **93.60%** | 93.60% | 26 min |
| G2 ResNet-18 @160 | ImageNet | 7 | 95.58% | **95.20%** | 95.19% | 10 min |

![CIFAR-10 confusion matrix](../ee6483/figures/fig_cifar_confusion.png)

![Per-class accuracy](../ee6483/figures/fig_cifar_perclass.png)

### Reading the results

1. From-scratch training is competitive here. G1 reaches 93.60% test accuracy in 25 epochs without any pre-training; the ImageNet-pretrained G2 reaches 95.20% in 7 epochs. The gap is only ~1.6 points: about five times smaller than the 7.5-point pre-training gap measured on cats vs. dogs at a comparable short budget (Section 7.2).
2. That is exactly what the learning curve in Section 7.3 predicts. Pre-training substitutes for data: it is worth most when the dataset is small (a pre-trained model on 500 cat/dog images reaches 93.6%, beating a from-scratch model trained on 20,000 images at 90.7%) and worth least when the dataset is large. CIFAR-10 has 45,000 training images of 32x32: small images are cheap to learn from, and the dataset is 2.25x larger than our cat/dog training set. The pre-training advantage shrinks accordingly. This cross-dataset consistency is stronger evidence than either experiment alone.
3. ImageNet pre-training is partly off-target here. Only 2 of the 10 CIFAR classes (cat, dog) exist in ImageNet-1k as such; the other 8 are coarse supersets (automobile, truck, ship, airplane) or absent (frog, deer). What transfers is the generic low/mid-level feature hierarchy, not a category prior: which is why the gain is real but modest.
4. The confusion structure is semantic, and it is the same structure we saw in Section 6. The single dominant off-diagonal pair is cat-dog (69 errors in each direction), i.e. exactly the confusion that dominated the Dogs vs Cats task: the same two classes are the hardest pair in both problems, which is a satisfying cross-task consistency check. The second tier is animal-shape confusion (cat→deer 22, frog→cat 21, bird→deer 18) plus a blue-background cluster (ship→airplane 20, airplane→ship 18, bird→airplane 18). The last group is informative: at 32x32 the vehicle classes are small objects on a sky/sea background, and the model does use that background as evidence: the shortcut we worried about in Section 1 is measurable here, just not for the animal classes.
5. Per-class accuracy tracks class difficulty, not class frequency (all classes are equally frequent here): automobile, ship and truck are easiest (96-98%), while cat, dog and bird are hardest (86-91%). That ordering is driven by within-class variability and object size, and it changes completely in Section 9, where frequency is manipulated instead: which is exactly the point of Section 9.

### What this tells us about the original problem

The CIFAR-10 experiment is a stress test of the claim made in Section 1: *the binding constraint on this project is data and transfer, not architecture.* Moving from 2 classes/224 px to 10 classes/32 px changes the output layer, the stem, the augmentation, the optimiser and the schedule: and the same recipe still works, with the same qualitative behaviour (pre-trained beats scratch, and the gap shrinks as data grows). The pipeline, not the specific model, is what generalises.

The one result that does *not* carry over is the size of the pre-training advantage, and that difference is itself informative: it is the same phenomenon (transfer substitutes for data) observed at two different points on the data axis.
