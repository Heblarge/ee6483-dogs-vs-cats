# 9. Class Imbalance

## 9.1 The problem

In Section 8 every class had 5,000 training images. Real datasets rarely look like that: a few "head" classes dominate the sample count while a long tail of "tail" classes has a handful of examples each. Training a softmax classifier on such data with plain cross-entropy fails in a specific, predictable way: the classifier's weights drift towards the head classes, the decision boundary is pushed into the tail, and *overall accuracy stays high* because predicting the head classes is usually right. Overall accuracy is therefore the wrong metric, a point returned to in Section 9.4.

The imbalance has two separable causes, which is what makes the remedies interesting:

1. Representation bias: the feature extractor sees few tail samples and learns a weaker representation for them.
2. Classifier bias: even with a perfectly balanced representation, the softmax logits are systematically larger for frequent classes because the loss is dominated by them.

Kang et al. [ICLR 2020] show that (2) dominates: if you freeze a representation trained on long-tailed data and only re-train the classifier with class-balanced sampling, you recover most of the gap. That finding is the basis of the decoupling method we test.

## 9.2 Constructing the imbalanced dataset

We use CIFAR-10 with an exponential long-tailed training distribution (the "CIFAR-10-LT" protocol):

$$n_i = n_{max}\cdot \rho^{-i/(C-1)},\qquad i=0,\dots,9,\quad \rho=100,\ n_{max}=5{,}000$$

which gives the per-class counts below. Only the *training* set is subsampled; validation and test keep the original balanced distribution, because the goal is to classify well on a balanced test set, not to reproduce the training prior.

| class | 0 airplane | 1 automobile | 2 bird | 3 cat | 4 deer | 5 dog | 6 frog | 7 horse | 8 ship | 9 truck |
|---|---|---|---|---|---|---|---|---|---|---|
| images | 4,500 | 2,997 | 1,797 | 1,077 | 646 | 387 | 232 | 139 | 83 | 50 |

Total training images: 11,908 (down from 45,000), imbalance ratio 4,500:50 = 90:1. The counts follow n_i = n_max x rho^(-i/9) with rho = 100 and n_max = 5,000, then clipped to the 4,500 images that remain per class after the 45,000/5,000 train/validation split; class 0 is the only one affected, which is why the realised ratio is 4,500:50 = 90:1 rather than the nominal 100:1.

The validation split (5,000, stratified) and the 10,000-image test set are untouched and balanced.

## 9.3 Five methods

Five methods are implemented so that the comparison spans the three families in the literature: data-level, loss-level and post-hoc.

The classifier is the CIFAR-ResNet-18 of Section 8 (model G1), retrained from scratch on the long-tailed training set. Only the loss, the sampling procedure or the decision rule changes between methods, so any difference in the results is attributable to the treatment of imbalance rather than to a change of architecture.

### 9.3.1 Cross-entropy baseline (CE)

Standard softmax cross-entropy, no correction. This is the control: whatever the imbalance costs, it costs here.

### 9.3.2 Re-sampling (data-level)

Draw each mini-batch with per-sample probability

$$p_j \propto \frac{1}{n_{c(j)}}$$

so that every batch is approximately class-balanced (`WeightedRandomSampler`, replacement=True). Tail images are therefore repeated many times per epoch.

*Why it should work:* it removes the classifier bias directly: the empirical label distribution seen by the loss becomes uniform.
*Known risk:* the tail classes are effectively oversampled by a factor of 10, so they overfit; this must be paired with strong augmentation. Our augmentation includes RandomCrop+flip+RandAugment+RandomErasing, which mitigates but does not eliminate it.

### 9.3.3 Class-balanced loss (loss-level)

Cui et al. [CVPR 2019] replace inverse frequency with the effective number of samples. With $E_{n} = (1-\beta^{n})/(1-\beta)$ and $\beta=0.9999$:

$$w_c \propto \frac{1}{E_{n_c}},\qquad \mathcal{L} = -\sum_c w_c\, y_c \log p_c$$

The intuition: as $n$ grows, additional samples add progressively less new information, so the "effective" size saturates and the weights should not diverge like $1/n$. With $\beta \to 1$ this recovers inverse-frequency weighting; with $\beta \to 0$ it recovers plain CE. We normalise the weights to mean 1 so the effective learning rate is unchanged.

*Why it should work:* it re-balances the loss without changing the data distribution, so no sample is repeated and there is no extra overfitting risk.
*Known risk:* with a 10:1 ratio the weights are mild; the gradient is still dominated by head-class batches early in training.

### 9.3.4 Decoupling / cRT (two-stage)

Kang et al. [ICLR 2020]: train normally for 80% of the schedule, then freeze the backbone, re-initialise the classifier, and re-train only the classifier with class-balanced sampling for the remaining 20%.

*Why it should work:* it follows the causal analysis of Section 9.1. the representation is nearly unbiased, the classifier is the biased part, so fix only the classifier and do not disturb the features.
*Implementation note:* the head is re-initialised (Kaiming) rather than merely re-tuned, because the existing head has already absorbed the prior; and the second stage uses a 10× smaller learning rate with a short cosine decay.

### 9.3.5 Logit adjustment (post-hoc)

Menon et al. [ICLR 2021]. Train with plain CE, then at *inference* shift the logits by the log-prior:

$$\hat{y} = \arg\max_c \big[f_c(x) + \tau \log \pi_c\big],\qquad \pi_c = \frac{n_c}{\sum_j n_j},\ \tau = 1$$

*Why it should work:* under a multinomial-logit model this is the Bayes-optimal correction for the balanced error rate, and it costs nothing at training time: no retraining, no extra hyper-parameter search.
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

1. Logit adjustment is the winner, and it requires no retraining. Applying `argmax f_c(x) - tau·log(pi_c)` to the *already trained* CE model raises test accuracy from 62.44% to 72.99% (+10.6 pp) and macro-F1 from 60.27% to 72.96%. The optimal tau selected on the validation split is exactly 1.0, which is what the theory predicts for a multinomial-logit model: a satisfying confirmation that the mechanism is the one the paper describes (a prior shift in the logits) rather than an accidental regularisation effect.
2. Re-sampling works but trades head for tail. +9.6 pp overall; head-class recall drops from 90.3% to 85.8% while the tail rises from 39.5% to 64.5%. This is the expected behaviour: repeating tail samples makes the effective training distribution uniform.
3. Decoupling works for the reason Kang et al. give. The training log shows it directly: validation accuracy is 52.6% at epoch 10, then the backbone is frozen and the head re-initialised at epoch 14, and one epoch later validation accuracy is 70.0%. The representation was already fine; the classifier was the biased part, and re-training it on a balanced stream recovered 18 points in 4 epochs. Final test accuracy 71.22%.
4. Class-balanced loss gives no gain at this schedule. It reaches 59.50%, *below* plain CE. The training curves explain why: at epoch 30 the training accuracy is still 59.95%, i.e. the model is severely under-trained. Two causes, both specific to our setup: (i) with weights normalised to mean 1 and `reduction='mean'` over weighted samples, head-class-dominated batches produce a *smaller* effective gradient, so the weighted loss converges more slowly than plain CE; (ii) the 30-epoch budget is far shorter than the 200+ epochs used in the original paper. Note that CB *does* re-balance: its few-class recall (63.1%) beats CE (39.5%): but it destroys the head (66.8% vs 90.3%) and never recovers overall accuracy. A shorter 18-epoch run (54.04%) shows the same pattern more strongly.

### Practical recommendation

Ordered by benefit per unit of effort:

1. Logit adjustment: one line at inference time, no retraining, +10.6 pp. Do this first, always.
2. Decoupling (cRT): one extra short training stage, +8.8 pp, and it is the most *interpretable* fix because it isolates the classifier as the biased component.
3. Re-sampling: trivial to implement, +9.6 pp, but it overfits the tail and needs strong augmentation to be safe.
4. Class-balanced loss: theoretically appealing, but in our hands it needs a much longer schedule to pay off; at 30 epochs it is a net loss.

The ranking is not the same as the ranking in the literature (where CB loss is standard), and the reason is the budget: methods that change the *loss scale* need time to converge, whereas methods that change only the *decision rule* are budget-free. That is a genuinely useful lesson for a project-sized experiment.

## 9.5 Honest limitations

- One seed per method, one imbalance ratio (rho=100). The long-tail literature sweeps rho in {10, 50, 100, 200} and averages over 3-5 seeds. Our ordering is robust to what we observe, but the magnitudes are not, and a single seed cannot separate a 1-point difference from noise.
- The whole study sits in an under-trained regime. Even the best method reaches 72.99%, whereas the CIFAR-10-LT literature reports ~77% for LDAM-DRW at rho=100 with 200-epoch schedules. Our comparison is internally consistent (four methods share one schedule and the class-balanced loss was given a longer one, as noted above), but the absolute numbers should not be compared with published ones.
- The backbone is trained from scratch. With an ImageNet-pretrained backbone the imbalance gap shrinks substantially, because the representation is already good: an interesting follow-up we did not have the budget for.
- Balanced accuracy here is macro-averaged recall on the balanced test set. We do not report the balanced error rate on the *training* prior, which is what logit adjustment is theoretically optimal for.
