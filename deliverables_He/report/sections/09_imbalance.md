# 9. Class Imbalance

## 9.1 The problem

In Section 8 every class had 5,000 training images. Real datasets rarely look like that: a few "head" classes dominate the sample count while a long tail of "tail" classes has a handful of examples each. Training a softmax classifier on such data with plain cross-entropy fails in a specific, predictable way: the classifier's weights drift towards the head classes, the decision boundary is pushed into the tail, and *overall accuracy stays high* because predicting the head classes is usually right. Overall accuracy is therefore the wrong metric, a point returned to in Section 9.4.

The imbalance has two separable causes, which is what makes the remedies interesting:

1. Representation bias: the feature extractor sees few tail samples and learns a weaker representation for them.
2. Classifier bias: even with a perfectly balanced representation, the softmax logits are systematically larger for frequent classes because the loss is dominated by them.

Kang et al. [ICLR 2020] show that (2) dominates: if you freeze a representation trained on long-tailed data and only re-train the classifier with class-balanced sampling, you recover most of the gap. That finding is the basis of the decoupling method we test.

## 9.2 Constructing the imbalanced dataset

We use CIFAR-10 with an exponential long-tailed training distribution (the "CIFAR-10-LT" protocol):

$$n_i = n_{max}\cdot \rho^{-i/(C-1)},\qquad i=0,\dots,9,\quad \rho=100,\ n_{max}=5000$$

which gives the per-class counts below. Only the *training* set is subsampled; validation and test keep the original balanced distribution, because the goal is to classify well on a balanced test set, not to reproduce the training prior.

| class | 0 airplane | 1 automobile | 2 bird | 3 cat | 4 deer | 5 dog | 6 frog | 7 horse | 8 ship | 9 truck |
|---|---|---|---|---|---|---|---|---|---|---|
| images | 4,500 | 2,997 | 1,797 | 1,077 | 646 | 387 | 232 | 139 | 83 | 50 |

Total training images: 11,908 (down from 45,000), imbalance ratio 4,500:50 = 90:1. The counts follow n_i = n_max x rho^(-i/9) with rho = 100 and n_max = 5,000, then clipped to the 4,500 images that remain per class after the 45,000/5,000 train/validation split; class 0 is the only one affected, which is why the realised ratio is 4,500:50 = 90:1 rather than the nominal 100:1.

The validation split (5,000, stratified) and the 10,000-image test set are untouched and balanced.

## 9.3 Five methods

Five methods are implemented so that the comparison spans the three families in the literature: data-level, loss-level and post-hoc.

The classifier is the CIFAR-ResNet-18 of Section 8 (model G1), retrained from scratch on the long-tailed training set. The architecture is the same for every method; what changes is the loss, the sampling procedure or the decision rule. One exception should be kept in mind when reading the results: the class-balanced loss row uses a longer, 30-epoch run (Appendix A.1), so it differs from the others in schedule as well as in loss, and is not a strictly one-variable comparison. The 18-epoch CB run is reported alongside it.

### 9.3.1 Cross-entropy baseline (CE)

Standard softmax cross-entropy, no correction. This is the control: whatever the imbalance costs, it costs here.

### 9.3.2 Re-sampling (data-level)

Draw each mini-batch with per-sample probability

$$p_j \propto \frac{1}{n_{c(j)}}$$

so that every batch is approximately class-balanced (`WeightedRandomSampler`, replacement=True). Tail images are therefore repeated many times per epoch.

*Why it should work:* it removes the classifier bias directly: the empirical label distribution seen by the loss becomes uniform.
*Known risk:* the sampler draws about 11,908 / 10 ≈ 1,190 images per class per epoch, so each of the 50 truck images is seen about 24 times per epoch (against roughly 0.26 times for an airplane image); the tail classes therefore overfit, and this must be paired with strong augmentation. Our augmentation includes RandomCrop+flip+RandAugment+RandomErasing, which mitigates but does not eliminate it.

### 9.3.3 Class-balanced loss (loss-level)

Cui et al. [CVPR 2019] replace inverse frequency with the effective number of samples. With $E_{n} = (1-\beta^{n})/(1-\beta)$ and $\beta=0.9999$:

$$w_c \propto \frac{1}{E_{n_c}},\qquad \mathcal{L} = -\sum_c w_c\, y_c \log p_c$$

The intuition: as $n$ grows, additional samples add progressively less new information, so the "effective" size saturates and the weights should not diverge like $1/n$. With $\beta \to 1$ this recovers inverse-frequency weighting; with $\beta \to 0$ it recovers plain CE. We normalise the weights to mean 1 so the effective learning rate is unchanged.

*Why it should work:* it re-balances the loss without changing the data distribution, so no sample is repeated and there is no extra overfitting risk.
*Known risk:* at a 90:1 ratio and $\beta=0.9999$ the weights are far from mild: $E_{4500}\approx 3624$ against $E_{50}\approx 49.9$, so a truck sample weighs about 73 times as much as an airplane sample. Head-class samples then contribute little gradient, which can slow the learning of the shared representation.

### 9.3.4 Decoupling / cRT (two-stage)

Kang et al. [ICLR 2020]: train normally for about 80% of the schedule, then freeze the backbone, re-initialise the classifier, and re-train only the classifier with class-balanced sampling for the rest. In our 18-epoch run the first stage is epochs 1–14 (identical to the CE baseline) and the classifier is re-trained in epochs 15–18.

*Why it should work:* it follows the causal analysis of Section 9.1. the representation is nearly unbiased, the classifier is the biased part, so fix only the classifier and do not disturb the features.
*Implementation note:* the head is re-initialised (Kaiming) rather than merely re-tuned, because the existing head has already absorbed the prior; and the second stage uses a 10× smaller learning rate with a short cosine decay.

### 9.3.5 Logit adjustment (post-hoc)

Menon et al. [ICLR 2021]. Train with plain CE, then at *inference* shift the logits by the log-prior:

$$\hat{y} = \arg\max_c \big[f_c(x) - \tau \log \pi_c\big],\qquad \pi_c = \frac{n_c}{\sum_j n_j}$$

Subtracting $\tau \log \pi_c$ raises the scores of rare classes (small $\pi_c$) relative to frequent ones. The strength $\tau$ is chosen on the validation split from $\{0, 0.25, 0.5, 0.75, 1.0, 1.5\}$ and then applied once to the test set (`ee6483/scripts/eval_logit_adjust.py`).

*Why it should work:* under a multinomial-logit model this is the Bayes-optimal correction for the balanced error rate, and it costs nothing at training time: no retraining, only a one-dimensional sweep over $\tau$ on cached validation logits.
*Known risk:* it assumes the training prior is the only source of bias; if the representation itself is skewed, the correction is incomplete.

## 9.4 Results

| Method | Family | Epochs | Overall acc | Macro-F1 | many (cls 0-2) | few (cls 6-9) |
|---|---|---|---|---|---|---|
| CE (baseline) | – | 18 | 62.44% | 60.27% | 90.3% | **39.5%** |
| Re-sampling | data | 18 | 72.02% | 72.00% | 85.8% | 64.5% |
| Class-balanced loss | loss | 30 | 59.50% | 59.48% | 66.8% | 63.1% |
| Decoupling (cRT) | two-stage | 18 | 71.22% | 71.14% | 79.6% | 71.0% |
| **Logit adjustment (tau=1.0)** | post-hoc | 18 (no extra training) | **72.99%** | **72.96%** | 81.4% | **72.0%** |

![Imbalance comparison](../ee6483/figures/fig_imbalance.png)

![CIFAR-10-LT validation curves per method](../ee6483/figures/fig_imbalance_curves.png)

### The pathology, stated numerically

Plain cross-entropy shows the textbook signature of long-tail training. Per-class recall on the balanced test set:

| | airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |
|---|---|---|---|---|---|---|---|---|---|---|
| samples seen | 4,500 | 2,997 | 1,797 | 1,077 | 646 | 387 | 232 | 139 | 83 | 50 |
| CE recall (%) | 94 | 98 | 78 | 72 | 71 | 53 | 63 | 49 | **24** | **22** |
| logit adj. recall (%) | 83 | 92 | 70 | 60 | 71 | 66 | 80 | 67 | **70** | **72** |

The two rarest classes collapse to 22-24% recall, barely above the 10% chance level, while the two most frequent classes reach 94-98%. The overall accuracy (62.4%) looks respectable and completely hides this. This is why every table in this section reports macro-F1 and per-class recall, not just accuracy.

### What each method does

1. Logit adjustment is the winner, and it requires no retraining. Applying `argmax f_c(x) - tau·log(pi_c)` to the *already trained* CE model raises test accuracy from 62.44% to 72.99% (+10.6 pp) and macro-F1 from 60.27% to 72.96%. The tau selected on the validation split from the six-value grid is 1.0, which coincides with the value the theory gives for a multinomial-logit model. Because the grid is coarse this is a consistency check rather than a confirmation, but it agrees with the mechanism the paper describes (a prior shift in the logits).
2. Re-sampling works but trades head for tail. +9.6 pp overall; head-class recall drops from 90.3% to 85.8% while the tail rises from 39.5% to 64.5%. This is the expected behaviour: repeating tail samples makes the effective training distribution uniform.
3. Decoupling works for the reason Kang et al. give. The training log shows it directly: epochs 1–14 are identical to the CE baseline and end at 61.0% validation accuracy; the backbone is then frozen and the head re-initialised, and after one epoch of class-balanced head re-training (epoch 15) validation accuracy is 70.0%. With the representation fixed, re-training only the classifier gained 8.9 points in one epoch, which is the pattern expected if the classifier, not the representation, carries most of the bias. Final test accuracy 71.22%. Table entries are rounded to one decimal, so the many/few averages quoted above can differ from recomputation on the rounded per-class recalls by up to 0.3 points.
4. Class-balanced loss gives no gain at this schedule. It reaches 59.50%, *below* plain CE. The training curves explain why: at epoch 30 the training accuracy is still 59.95%, i.e. the model is severely under-trained. Two causes, both specific to our setup: (i) with weights normalised to mean 1 and `reduction='mean'` over weighted samples, head-class-dominated batches produce a *smaller* effective gradient, so the weighted loss converges more slowly than plain CE; (ii) the 30-epoch budget is far shorter than the 200+ epochs used in the original paper. Note that CB *does* re-balance: its few-class recall (63.1%) beats CE (39.5%): but it destroys the head (66.8% vs 90.3%) and never recovers overall accuracy. A shorter 18-epoch run (54.04%) shows the same pattern more strongly.

### Practical recommendation

Ordered by benefit per unit of effort:

1. Logit adjustment: one line at inference time, no retraining, +10.6 pp. In this setting it is the cheapest first step to try.
2. Decoupling (cRT): one extra short training stage, +8.8 pp, and it is the most *interpretable* fix because it isolates the classifier as the biased component.
3. Re-sampling: trivial to implement, +9.6 pp, but it overfits the tail and needs strong augmentation to be safe.
4. Class-balanced loss: theoretically appealing, but in our hands it needs a much longer schedule to pay off; at 30 epochs it is a net loss.

The ranking is not the same as the ranking in the literature (where CB loss is standard), and the reason is the budget: methods that change the *loss scale* need time to converge, whereas methods that change only the *decision rule* are budget-free. That is a genuinely useful lesson for a project-sized experiment.

## 9.5 Honest limitations

- One seed per method, one imbalance ratio (rho=100). The long-tail literature sweeps rho in {10, 50, 100, 200} and averages over 3-5 seeds. The ordering and the magnitudes are therefore single-run observations: the 9–10 point gains over CE are large enough to be unlikely to reverse, but a single seed cannot separate the 1–2 point differences among logit adjustment, re-sampling and decoupling from noise, and none of the ranking has been tested at other imbalance ratios.
- The whole study sits in an under-trained regime. Even the best method reaches 72.99%, whereas the CIFAR-10-LT literature reports ~77% for LDAM-DRW at rho=100 with 200-epoch schedules. Our comparison is internally consistent (four methods share one schedule and the class-balanced loss was given a longer one, as noted above), but the absolute numbers should not be compared with published ones.
- The backbone is trained from scratch. With an ImageNet-pretrained backbone the imbalance gap shrinks substantially, because the representation is already good: an interesting follow-up we did not have the budget for.
- Balanced accuracy here is macro-averaged recall on the balanced test set, i.e. one minus the balanced error rate, which is the quantity logit adjustment is designed to optimise. Because the test set is balanced, it coincides with overall accuracy.
