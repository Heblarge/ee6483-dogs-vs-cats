# 6. Error Analysis

## 6.1 What can and cannot be analysed

A prediction can only be called correct or incorrect where a label exists. The test set is unlabelled, so the analysis is split between the validation set, where error rates can be measured, and the test set, where only a qualitative inspection is possible.

| Sections | Data | Labels | What it supports |
|---|---|---|---|
| 6.2 to 6.6 | validation set (5,000 images) | Yes | quantitative error rate, failure taxonomy, Grad-CAM, robustness |
| 6.7 | test set (500 images) | No | qualitative only: confidence distribution and inspection of extreme cases |

Statements about whether a particular test prediction is correct are manual judgements and are identified as such.

## 6.2 How often, and in which direction

| | predicted cat | predicted dog |
|---|---|---|
| **true cat** | 2,482 (99.28%) | 18 (0.72%) |
| **true dog** | 32 (1.28%) | 2,468 (98.72%) |

50 errors out of 5,000 (99.00% accuracy) for the single-seed model (seed 42); the three-seed ensemble used for the submission makes 39 errors on the same set, i.e. ensembling removes about a fifth of them. The errors are asymmetric: the model confuses dogs for cats about 1.8× more often than the reverse. Two plausible causes, both consistent with the case studies below:

1. The "dog" class has far more visual diversity (340+ breeds, from 1 kg to 90 kg), so its decision region is broader and its boundary is fuzzier on the cat side.
2. Several dog breeds are small, long-haired and round-faced (Pomeranian, Shih Tzu, Bichon): i.e. they sit inside the cat region of feature space. The opposite direction has no strong analogue.

![Confusion matrix](../ee6483/figures/fig_confusion_dogs.png)

## 6.3 Failure taxonomy

All 50 validation errors were inspected (the ten most confident are shown below, five in each direction; the complete set is in `figures/fig_errors_all_1.png` to `fig_errors_all_5.png`) and each was assigned to a single dominant cause. The categories are not perfectly disjoint, since an image can be both occluded and background-driven, so the rule was "what would have to change for this image to be classified correctly?".

| # | Failure mode | Count | Share | Representative example |
|---|---|---|---|---|
| 1 | **Face not visible**: rear view, curled up asleep, head turned away, or an extreme close-up on a body part | **18** | 36% | `dog.3519` (white dog from behind, p=0.93) |
| 2 | **Occlusion or unusual framing**: wire mesh, held in a hand, inside a crate | **15** | 30% | `cat.9897` (cat behind a chain-link fence, p=0.94) |
| 3 | **Background / scene context contradicts the class** | 5 | 10% | `cat.6734` (black cat walking on grass, p=0.90) |
| 4 | **Multi-animal image or strong distractor objects** | 5 | 10% | `cat.7020` (cat lying next to a beagle, p=0.90) |
| 5 | **Label noise, non-photographic content, or severely degraded image** | 5 | 10% | `cat.3072` (group photo of five people, no animal at all) |
| 6 | **Cat-like small dog / puppy-kitten ambiguity** | 2 | 4% | `dog.4030` (miniature pinscher, p=0.88) |

![The ten most confident validation errors, five in each direction (true label, prediction and probability annotated)](../ee6483/figures/fig_errors_balanced.png)

Four observations:

- Two thirds of the errors (33/50) are "the animal is not presented the way the training set presents it": no face, or wrapped in a fence/hand. The model is not confused about *what* a cat or dog is; it is missing the specific view it has learned to rely on. This is a coverage problem, and it is the single most useful conclusion of the error analysis.
- Category 1 is the largest source of error, and it is face-driven. Grad-CAM (Section 6.6) confirms that on correct predictions the evidence is concentrated on the head. That prior is usually right, which is why accuracy is 99%, but it has no fallback when the head is hidden.
- Category 4 is an annotation problem, not a model problem. With a single-label target, an image containing both animals has no correct answer; whichever label is assigned, the model is "wrong" half the time by construction. This sets a floor on achievable accuracy that no architecture can remove.
- Category 5 is irreducible, and Section 2 explains why. `cat.3072` is a photograph of five people at a trade-show booth labelled "cat"; `cat.11565` is the "PHOTO UNAVAILABLE" placeholder that also appears three times under `dog` (Section 2.2); `dog.1568` is a tabby cat held by a person, labelled "dog". No model can fit these, since the same pixels carry two different targets, so they cap achievable accuracy on this validation set regardless of architecture. This is the concrete cost of the 6 placeholder images found by the pHash audit.

Category 3 is the shortcut learning discussed in Section 1: a black cat walking on grass is predicted "dog" with 0.90 confidence. The model has learned a genuine correlation in the data (dogs photographed outdoors more often), and it exploits that correlation when the animal itself is ambiguous.

## 6.4 Case study: the white dog photographed from behind

*Third panel of the grid above:* a small white dog photographed from behind, predicted cat with probability 0.93.

Why the model is wrong. There is no face in the image. The remaining evidence is (i) a compact, rounded body, (ii) long, soft, light-coloured fur, and (iii) a curled tail. Every one of these is also a property of a long-haired cat. The model has no feature that discriminates on the basis of *tail shape* or *ear placement* alone, because in the training set those cues are almost always co-present with a face, so the network never had to learn them independently.

What this reveals about the model's strength. On the vast majority of images the face is visible, and there the model is essentially perfect: it has learned a genuinely discriminative, high-level representation (Section 6.6). The failure is a *coverage* failure, not a representation failure: the training distribution contains few rear-view dogs, so the model is being asked to extrapolate.

What would fix it. More aggressive cropping (`RandomResizedCrop` with a lower scale bound) would present more partial animals during training; alternatively a part-based model or an attention mechanism that explicitly localises head and tail would decouple the two cues. Neither is a hyper-parameter: this is a data-coverage problem.

## 6.5 Case study: the cat behind a chain-link fence

*Second panel of the same grid:* an orange-and-white cat behind a chain-link fence, predicted dog with probability 0.94.

Why the model is wrong. The mesh introduces a strong, regular high-frequency pattern over the entire animal. Two mechanisms may compound:

1. The pattern corrupts the local texture statistics that the network uses to describe fur: the very feature that separates a cat from a fluffy dog.
2. Enclosures (kennels, yards, crates) are plausibly more frequent in dog photographs in this dataset, so the *context* may also vote "dog". We state this as a hypothesis: we did not run the counterfactual experiment (the same cat photographed with and without the fence) that would be needed to separate the two.

What this reveals about the model's weakness. A human resolves this instantly by integrating the visible face through the mesh. The network's receptive field at the deepest layer covers the whole image, so it *could* in principle do the same, but nothing in the training objective forces it to be robust to structured occlusion: `RandomErasing` (which we use) inserts a *solid* rectangle, not a *high-frequency* pattern, so it does not train this invariance.

Consistency with the robustness test. This is exactly what the corruption experiment in Section 6.6 shows quantitatively: blur, which also destroys high-frequency texture, costs 12.7 points, while a 25% solid occlusion costs only 0.4 points. The model is texture-driven, and any corruption that attacks texture attacks it.

## 6.6 Grad-CAM and the robustness test

Grad-CAM [Selvaraju et al., ICCV 2017] weights the last convolutional feature maps by the gradient of the predicted class score, producing a coarse localisation of the evidence the model used.

![Grad-CAM, correct predictions](../ee6483/figures/gradcam_correct.png)

For correct predictions the activation is concentrated on the head and face in essentially every case, and not on the background. This is the strongest evidence we have that the model learned the animal rather than the scene: the concern raised in Section 1.1.2 about background shortcut learning is, for this model, largely unfounded on easy images.

![Grad-CAM, incorrect predictions](../ee6483/figures/gradcam_wrong.png)

For incorrect predictions the pattern is diagnostic, and it splits by direction:

- Both cat-to-dog errors are cats photographed on grass (0.92 and 0.90). In each case the activation covers the animal but also the surrounding lawn, so the scene contributes evidence; this is the background cue discussed in Section 1 and quantified as failure category 3 in Section 6.3.
- Both dog-to-cat errors are dogs whose face is not visible: one is held against a person's body (0.94), the other is photographed from behind (0.93). The activation falls on the back, the fur and the person's hands rather than on a head, which is the coverage failure described in Sections 6.4 and 6.5.

Robustness test. We apply eight deterministic perturbations to the validation images and re-evaluate without retraining:

| Perturbation | val acc (%) | Δ vs clean |
|---|---|---|
| none (clean) | 99.00 | – |
| Gaussian blur σ=2 | 95.26 | −3.7 |
| Gaussian blur σ=4 | 86.34 | **−12.7** |
| brightness ×1.6 | 98.48 | −0.5 |
| brightness ×0.45 | 98.74 | −0.3 |
| rotation 30° | 97.94 | −1.1 |
| 25% central occlusion | 98.56 | −0.4 |
| Gaussian noise σ=25 | 94.58 | −4.4 |

![Robustness](../ee6483/figures/fig_robustness.png)

Three conclusions:

1. Blur is by far the most damaging (−12.7 pp). Accuracy under heavy blur collapses to 86%, i.e. worse than a small CNN trained on clean data would score on clean images. Since blur removes exactly the high-frequency band that encodes fur texture, this is direct evidence that the decision is texture-driven rather than shape-driven.
2. Occlusion is almost free (−0.4 pp): the RandomErasing augmentation did its job. This is a case where a training-time choice produced a measurable robustness benefit, which is a point in favour of the augmentation policy in Section 2.
3. Colour is nearly irrelevant (−0.3 to −0.5 pp). Combined with conclusion 1, the model's evidence hierarchy is: face geometry > fur texture > colour/context. That ordering explains both case studies above.

## 6.7 The test set: qualitative inspection

![Test-set predictions of the main-study model (M3 ×3 + TTA), four least confident and two most confident per class. Labels are M3's predictions, not ground truth: id=374, labelled dog, actually shows two cats and is an M3 error that the final submitted ensemble corrects](../ee6483/figures/fig_test_confidence.png)

The test set is unlabelled, so the labels shown are the model's predictions and any judgement about whether a prediction is correct is a visual assessment by the authors. Sections 6.7 and 6.8 describe the main-study ensemble (M3 ×3 + TTA); differences in the final submitted ensemble (Section 5.3) are noted where they occur.

- The four least confident cat predictions are a white cat on a patterned blanket (id=475, 0.68), a black cat on a rug (id=242, 0.72), a cat on the back of a chair (id=306, 0.75) and a dark tabby in a tiled room (id=46, 0.80). Each is a low-contrast or cluttered scene where a human would also hesitate, so the confidence is informative rather than miscalibrated.
- The four least confident dog predictions include the two "No Photo Available" placeholders (ids 464 and 328, 0.51 and 0.52), a small dog on a carpet (id=49, 0.60) and an image of a cat bed (id=374, 0.62). On closer inspection id=374 shows two cats curled up inside the bed, so M3 is wrong here: the cat-bed context and the tangled bodies, with neither face fully visible, pull it towards dog. The final ensemble corrects it (p(dog) = 0.06), as it does id=453, a Siamese cat that M3 also labels dog. The placeholders are the lowest-confidence inputs in the whole test set, which is the behaviour a closed-set classifier should exhibit on an input that contains no animal: it hedges rather than committing.
- The four most confident predictions are unambiguous: kittens on a cushion (id=200, 0.99), cats in a crate (id=176, 0.99), a black dog with a white puppy (id=157, 0.99) and a spaniel outdoors (id=323, 0.98). High confidence coincides with clear, frontal, well-lit subjects.
- The predicted class distribution is 261 dog / 239 cat (52/48) for M3 and 259 / 241 for the final ensemble, consistent with a roughly balanced test set. A reversed label convention or a lexicographic id sort would produce a very different distribution, so this is weak but real evidence that the submission pipeline is correct.

## 6.8 Two test-set cases: one strength, one weakness

The test set carries no labels, so the two cases below are chosen for what they reveal about the model rather than for a score. In both, the judgement that the prediction is right or wrong is a visual one.

*Case 1, a strength: two dogs of very different size (id=157, predicted dog, p=0.99).* The image contains a large black dog and a small white puppy side by side, at different scales and with different coat colours. The model assigns 0.99 to dog. Two properties of the training set are being handled at once: the presence of two animals, which would be ambiguous under a single-label target, and a scale ratio of roughly ten to one, which is the largest within-class variation in the dataset. This is the behaviour one would want from a system deployed on user photographs, where multiple animals and mixed sizes are common.

*Case 2, a weakness: an image that contains no animal (id=464, predicted dog, p=0.51).* The input is a "No Photo Available" placeholder graphic. The model returns "dog" with 0.51 confidence, which is the lowest value in the whole test set but is still a commitment. The weakness is structural rather than a training failure: a two-way softmax must put its mass somewhere, so an input outside the label space cannot be rejected. A maximum-softmax threshold or an explicit rejection class would fix it, at the cost of a validation set containing out-of-distribution images to calibrate the threshold (Section 6.9). The stronger final ensemble makes this weakness worse rather than better: it labels both placeholders dog with p ≈ 0.88 (ids 464 and 328), so it no longer hedges on them, and a confidence threshold calibrated for M3 would not transfer to it. Higher in-distribution accuracy does not imply better behaviour on inputs outside the label space.

The pair also illustrates where the model's remaining error budget lies. It is confident and correct when the animal is visible and well framed, however many animals or scales are present; it is uncertain, but still forced to answer, when the input is not an animal at all.

## 6.9 Limitations: closed-set assumption and calibration

Two limitations follow from the analysis above.

1. The classifier is closed-set and cannot return "neither". Images that contain no animal are still assigned a label. The six placeholder graphics are the clearest case: they receive the two lowest confidences in the whole test set (0.51 and 0.52), so the model does hedge, but a maximum-softmax threshold would be a more reliable rejection rule than reading the confidence by eye. The standard remedies, such as thresholding the maximum softmax probability [Hendrycks & Gimpel, ICLR 2017], OpenMax [Bendale & Boult, CVPR 2016] or an energy score, are post-hoc and cost nothing to add, but they need a validation set containing out-of-distribution images to calibrate the threshold, which this dataset does not provide.
2. Confidence is informative but not calibrated. Mean maximum-softmax confidence is 0.94 on the test set and 0.99 on correctly classified validation images, against an accuracy of 0.99; the two coincide on the validation set, so the confidences can be read as rough probabilities there, but nothing guarantees this on new data. Temperature scaling [Guo et al., ICML 2017] fitted on the validation set would make the calibration explicit. It was not applied because it does not change the argmax and therefore does not change the submission.

