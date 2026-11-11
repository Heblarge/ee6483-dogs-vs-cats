# 6. Error Analysis

## 6.1 What can and cannot be analysed

A prediction can only be called correct or incorrect where a label exists. The test set is unlabelled, so the analysis is split between the validation set, where error rates can be measured, and the test set, where only a qualitative inspection is possible.

| Sections | Data | Labels | What it supports |
|---|---|---|---|
| 6.2 to 6.6 | validation set (5,000 images) | Yes | quantitative error rate, failure taxonomy, Grad-CAM, robustness |
| 6.7 to 6.8 | test set (500 images) | No | qualitative only: submitted-ensemble confidence distribution and inspection of extreme cases |

Statements about whether a particular test prediction is correct are manual judgements and are identified as such.

## 6.2 How often, and in which direction

| | predicted cat | predicted dog |
|---|---|---|
| **true cat** | 2,482 (99.28%) | 18 (0.72%) |
| **true dog** | 32 (1.28%) | 2,468 (98.72%) |

50 errors out of 5,000 (99.00% accuracy) for the single-seed reference model (M3, seed 42); the three-seed ResNet-18 reference ensemble makes 39 errors on the same set, i.e. ensembling removes about a fifth of them. The submitted backbone ensemble makes 15 errors (Section 5.3). Sections 6.2–6.6 use the single model because attribution methods such as Grad-CAM require one network; Sections 6.7–6.8 switch explicitly to the submitted ensemble for test-set inspection. The single-model errors are asymmetric: it confuses dogs for cats about 1.8× more often than the reverse. Two plausible causes, both consistent with the case studies below:

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
- Category 5 is largely irreducible, and Section 2 explains part of why. `cat.11565` is the "PHOTO UNAVAILABLE" placeholder that also appears three times under `dog` (Section 2.2), so the same graphic carries two different targets; `cat.3072` is a photograph of five people at a trade-show booth labelled "cat"; `dog.1568` is a tabby cat held by a person, labelled "dog". The last two are label noise rather than contradictory duplicates: a model could still output the given label, but only by chance or by fitting the noise, not by recognising the animal. These images therefore limit the accuracy a sound classifier can be expected to reach on this validation set, regardless of architecture. This is the concrete cost of the 6 placeholder images found by the pHash audit.

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

Consistency with the robustness test. Section 6.6 shows a 12.7-point loss under heavy blur and a 0.4-point loss under a central square mask whose side is 25% of the original image's shorter dimension. This supports sensitivity to blur under the tested conditions. It does not isolate texture from shape cues or establish the cause of the fence-related errors.

## 6.6 Grad-CAM and the robustness test

Grad-CAM [Selvaraju et al., ICCV 2017] weights the last convolutional feature maps by the gradient of the predicted class score, producing a coarse localisation of the evidence the model used.

![Grad-CAM, correct predictions](../ee6483/figures/gradcam_correct.png)

For correct predictions the activation is concentrated on the head and face in essentially every case, and not on the background. This is the strongest evidence we have that the model learned the animal rather than the scene: the concern raised in Section 1.1.2 about background shortcut learning is, for this model, largely unfounded on easy images.

![Grad-CAM, incorrect predictions](../ee6483/figures/gradcam_wrong.png)

For incorrect predictions the pattern is diagnostic, and it splits by direction:

- Both cat-to-dog errors are cats photographed on grass (0.92 and 0.90). In each case the activation covers the animal but also the surrounding lawn, so the scene contributes evidence; this is the background cue discussed in Section 1 and quantified as failure category 3 in Section 6.3.
- Both dog-to-cat errors are dogs whose face is not visible: one is held against a person's body (0.94), the other is photographed from behind (0.93). The activation falls on the back, the fur and the person's hands rather than on a head, which is the coverage failure described in Sections 6.4 and 6.5.

Robustness test. We apply seven deterministic perturbations (plus the clean condition) to the validation images and re-evaluate without retraining:

| Perturbation | val acc (%) | Δ vs clean |
|---|---|---|
| none (clean) | 99.00 | – |
| Gaussian blur σ=2 | 95.26 | −3.7 |
| Gaussian blur σ=4 | 86.34 | **−12.7** |
| brightness ×1.6 | 98.48 | −0.5 |
| brightness ×0.45 | 98.74 | −0.3 |
| rotation 30° | 97.94 | −1.1 |
| Central square mask, side = 25% of original short side | 98.56 | −0.4 |
| Gaussian noise σ=25 | 94.58 | −4.4 |

The mask is applied before resizing and centre cropping. Its side length is approximately 0.25 × min(height, width), with integer rounding. It covers approximately 6.25% of a square original image, and a smaller proportion of a rectangular original; the fraction of the final model input depends on the subsequent crop. This experiment did not test a mask covering 25% of the image area.

![Robustness of the single-seed reference model. The occ25 label denotes a central square mask with side length 25% of the original image's shorter dimension, not 25% of its area](../ee6483/figures/fig_robustness.png)

Three conclusions:

1. Heavy blur causes the largest observed drop (−12.7 pp). This is consistent with reliance on fine visual detail, but blur also affects edges and shape information; these measurements alone cannot distinguish texture reliance from shape reliance.
2. The tested central mask causes a smaller drop (−0.4 pp). RandomErasing is a possible contributor, but no matched robustness comparison with RandomErasing disabled was performed, so its causal contribution is unknown. The small mask also limits how broadly this result can be generalised to occlusion.
3. The two brightness changes cause drops of about 0.3–0.5 pp. This supports robustness to those particular illumination changes. It does not establish robustness to hue or saturation changes, or a ranking of colour, texture and shape cues.

## 6.7 The test set: qualitative inspection

![Submitted six-model ensemble: four least-confident and two most-confident predictions per class on the unlabelled test set. Perceptual near-duplicates are suppressed; labels are predictions, not ground truth](../ee6483/figures/fig_test_confidence.png)

The test set is unlabelled, so the labels shown are the submitted ensemble's predictions and any judgement about whether a prediction appears plausible is a visual assessment by the authors. The panel is generated directly from `submission_backbone_ensemble_probs.csv`; it therefore describes the same six checkpoints and horizontal-flip TTA configuration as `submission.csv`. Perceptual near-duplicates are suppressed so that a repeated image cannot fill several slots. The second test-set placeholder (id=464, dog 0.88) is not shown because it is the fifth least-confident dog prediction, just above id=328, not because of this filter.

- The four least-confident cat predictions are a black-and-white cat behind cage bars with overlaid text (id=216, 0.84), a black cat on a dark rug (id=242, 0.85), a black-and-white cat held up by a hand in front of a chain-link fence, with a numbered tag (id=159, 0.91), and a grey cat on a strongly patterned red blanket (id=30, 0.93). Occlusion, low contrast and texture-heavy backgrounds are among the failure factors identified on the labelled validation set.
- The four least-confident dog predictions are a black puppy held in a hand (id=313, 0.84), a puppy sitting in a cage (id=261, 0.86), a low-resolution, pixelated close-up of a dog's face (id=302, 0.86) and a "No Photo Available" placeholder (id=328, 0.87). The first three predictions are visually plausible; the placeholder exposes the closed-set limitation because the model must choose cat or dog even when neither is present.
- The most-confident examples remain visually straightforward: two cat images (ids 176 and 198, both 0.96) and two dog images (ids 254 and 243, both 0.96). One dog image contains several animals behind a fence, showing that the ensemble can remain stable under multiple subjects and structured occlusion.
- The submitted prediction distribution is 259 dog / 241 cat (52/48). This is a descriptive count, not a check of image-to-label alignment: permuting predictions among ids leaves the count unchanged, and reversing labels simply swaps the two counts. Alignment is checked using numeric image ids and the explicit cat=0, dog=1 convention, as described in Section 5.3.

## 6.8 Two test-set cases: one strength, one weakness

The test set carries no labels, so the two cases below are chosen for what they reveal about the model rather than for a score. In both, the judgement that the prediction is right or wrong is a visual one.

*Case 1, a strength: several dogs behind a fence (id=243, predicted dog, p=0.96).* The image contains multiple dogs at different depths and scales, with a chain-link fence crossing the scene. The submitted ensemble remains confident despite multiple subjects, structured occlusion and a strong background pattern—the same factors that cause some single-model validation errors. This is the behaviour one would want from a system deployed on user photographs, where framing is rarely as clean as the training examples.

*Case 2, a weakness: an image that contains no animal (id=328, predicted dog, p=0.87).* The input is a "No Photo Available" placeholder graphic. The weakness is structural rather than a training failure: a two-way softmax must put its mass somewhere, so an input outside the label space cannot be rejected. A maximum-softmax threshold or explicit rejection class would address this, but would require a validation set containing out-of-distribution images to calibrate the threshold (Section 6.9). The relatively high confidence shows that higher in-distribution accuracy does not imply reliable behaviour outside the label space.

These two examples illustrate visually plausible recognition of dogs alongside a confident assignment to an input containing no animal. They do not establish a test error rate, and the placeholder shows why confidence alone is insufficient to identify inputs outside the model's label space.

## 6.9 Limitations: closed-set assumption and calibration

Two limitations follow from the analysis above.

1. The classifier is closed-set and cannot return "neither". Images that contain no animal are still assigned a label. The two test-set placeholders, which show the same graphic (ids 464 and 328, a "No Photo Available" graphic that does not occur in train or validation, unlike the six of Section 2.2) are the clearest case: both are predicted dog with probabilities 0.88 and 0.87. Standard remedies such as maximum-softmax thresholding [Hendrycks & Gimpel, ICLR 2017], OpenMax [Bendale & Boult, CVPR 2016] or an energy score are post-hoc, but they need a validation set containing out-of-distribution images to calibrate the rejection threshold, which this dataset does not provide.
2. Confidence is informative but not calibrated. Mean maximum-softmax confidence is 0.95 on the unlabelled test set, but without labels no calibration error can be measured there. Temperature scaling fitted on the labelled validation set would make calibration explicit. It was not applied because it does not change the argmax and therefore does not change the submission.

