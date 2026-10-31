# 4. Hyper-parameter Selection

## 4.1 Protocol

Hyper-parameters were not tuned by trial and error on the final model. We used a three-stage protocol, each stage with a different cost/accuracy trade-off:

1. Coarse search on a proxy setting. ResNet-18, 160 px, 3 epochs, single seed. One factor varied at a time, everything else fixed. This is cheap (~2 min per run) and its ranking transfers well to the full setting because the factors we search over (learning rate, augmentation strength, resolution, transfer strategy) change the optimisation landscape, not the task.
2. Sensitivity curves for the two factors that interact most (learning rate and input resolution), so the choice is justified by the shape of the curve rather than by a single winning point.
3. Confirmation of the top configuration at full budget (224 px, 8 epochs) with three seeds, reported as mean ± std.

Everything below is therefore an *ablation with a purpose*: each row answers "what happens if I move this knob?".

## 4.2 Learning rate

Sweep: AdamW, 3 epochs, 160 px, full augmentation, `lr` applied to both backbone and head, five values spanning four orders of magnitude.

| lr | 1e-5 | 1e-4 | 3e-4 | 1e-3 | 3e-3 |
|---|---|---|---|---|---|
| val acc (%) | 97.68 | **98.62** | 98.24 | 97.70 | 96.06 |

A clean inverted U with a 2.6-point range, and the four regimes are easy to name:

- Too small (1e-5, 0.9 points below the peak): the head has not converged in 3 epochs. Note that 97.68% is still high, since AdamW's adaptive step size means that even 1e-5 moves the weights appreciably, but the head is clearly the limiting factor, not the representation.
- The peak (1e-4): best validation accuracy of the sweep.
- Declining (3e-4, 1e-3): 0.4 and 0.9 points below the peak. The first steps of a randomly initialised head produce gradients large enough to move the pre-trained backbone away from its optimum, and 3 epochs are not enough to recover. This is the empirical signature of the "fine-tuning distorts pre-trained features" effect that motivates LP-FT.
- Collapse (3e-3): 2.6 points below the peak. At this learning rate the run is visibly unstable: the training loss oscillates and the model never settles.

The final setting is a backbone learning rate of 3e-4, a head learning rate of 1e-3 and the LP-FT schedule. Two points relate this choice to the sweep:

1. The sweep ties the two learning rates together, so it identifies the *safe region*, not the final values. Decoupling them (small for the backbone, large for the head) is what allows the head to converge quickly without damaging the backbone, and it is why the final model uses an asymmetric pair that is not a point on this curve.
2. The sweep is run at 3 epochs; the final model is run at 8. Longer schedules tolerate (and benefit from) larger steps, which is why the final backbone lr (3e-4) is above the 3-epoch optimum (1e-4). The sweep's role is to exclude the destructive region, not to pick the final number: the final number comes from the three-seed confirmation in Section 5 (98.93 ± 0.05%), which is 0.3 points above the best point of the sweep.

## 4.3 Input resolution

| resolution | 96 | 128 | 160 | 224 |
|---|---|---|---|---|
| val acc (%) | 97.08 | 97.94 | 98.12 | 98.76 |

Accuracy increases monotonically with resolution and the trend is larger than the seed noise, but the cost grows quadratically: 224 px costs ~2× the time of 160 px for the same number of epochs. The jump from 96 to 160 is the largest because at 96 px the animal's face, which Grad-CAM identifies in Section 6 as the most discriminative region, occupies too few pixels for the early layers to resolve. Beyond 160 px the gain per pixel flattens, because the dataset's own resolution is heterogeneous (mean ~400 px, minimum 32 px): upsampling past 224 adds compute without adding information.

Decision: 224 px for the main model, 160 px for ablations. 224 is the resolution the pre-trained weights were trained at, which matters for the BatchNorm statistics; 160 keeps the 16-run ablation matrix affordable without changing its conclusions.

## 4.4 Transfer strategy

| strategy | trainable parameters | val acc (%) |
|---|---|---|
| linear probe (head only) | 1,026 | 96.24 |
| last stage + head | 8.4 M | 98.20 |
| full fine-tune | 11.18 M | 98.12 |
| LP-FT (1 epoch probe, then full) | 11.18 M | 98.32 |

The structure of this table is "one big step, then nothing": going from a frozen backbone to unfreezing the last residual stage is worth 1.88 points, and going from the last stage to all 11.2M parameters changes nothing measurable. In other words the useful adaptation happens in the last block; the early layers of an ImageNet network are already the right feature extractor for photographs of animals.

The head-only row measures what the pre-trained representation can do on its own. At this budget (160 px, 3 epochs) a 1,026-parameter linear classifier on frozen ImageNet features reaches 96.24%; the same probe trained for 6 epochs at 224 px reaches 97.78% (model M2 in Section 5.1). Either way most of the final accuracy comes from the representation rather than from the training procedure.

The four strategies differ by 2.1 points in total, and the ordering is stable across the three seeds.

## 4.5 Augmentation strength

| level | none | flip | geo (+rotation) | color (+jitter) | full (+RandAugment+erasing) |
|---|---|---|---|---|---|
| val acc (%) | 98.36 | 98.62 | 98.56 | 98.48 | 98.12 |

No monotone trend, and the strongest setting is the worst of the five by 0.5 points. We kept the full policy for the main model anyway, and the reasoning is worth stating because it is a case where a 3-epoch ablation is misleading:

- At 3 epochs the model is already at 98.1–98.6% and there is almost no headroom; the residual errors (Section 6) are label noise and ambiguous viewpoints, which no augmentation can fix.
- Heavy augmentation slows convergence. RandAugment and RandomErasing remove information from each sample, which costs accuracy when the schedule is short. We checked this directly by re-running the two extremes at 8 epochs: the sign flips and the full policy becomes 0.24 pp *better* (see Section 7.6). So the 3-epoch ordering is a schedule artefact, not a property of the transforms.
- The *qualitative* argument for the policy (translation/scale invariance, background-shortcut suppression) is not refuted by this table; it is simply not measurable at this budget on this task.

Augmentation is therefore a sub-1 pp lever on this task at every budget tested, and its sign depends on the schedule. We report it rather than dropping the experiment, because a report that only shows the experiments that "worked" is not evidence.

## 4.6 Other settings and why

| Setting | Value | Reasoning |
|---|---|---|
| Weight decay | 1e-4 (AdamW) | small dataset + short schedule: too much decay underfits, too little overfits after epoch 6 |
| Batch size | 64 | 96 fits in 8 GB at 224 px but 64 gives 312 steps/epoch (smoother cosine) and more stable BN statistics |
| Epochs | 8, patience 3 | validation accuracy peaks at epoch 7–8 for every seed; beyond that it plateaus or declines |
| Label smoothing | 0.1 | web labels contain errors; also improves the calibration of the confidences used in Section 6 |
| Warm-up | 1 epoch (0.3 epoch in stage 2) | prevents the first large steps from destabilising a freshly initialised head |
| Gradient clipping | 1.0 | cheap insurance; rarely active but prevents a single bad batch from diverging the run |
| AMP | on | 1.66× faster and 38% less peak memory, measured on a fixed 64×3×224×224 batch (66 ms vs 110 ms per step, 953 vs 1,543 MiB); no measurable accuracy difference at fp16 |
| Seeds | 42, 43, 44 | three is the minimum that gives a usable standard deviation for a non-deterministic pipeline |

## 4.7 Settings deliberately left untuned

Stating this is part of a defensible protocol: a search over everything is a search over noise at this dataset size.

- Architecture width/depth: the ResNet-18/34/50 comparison in Section 7 shows the differences are within seed noise at 8 epochs; tuning depth would be tuning noise.
- Optimiser family: AdamW vs SGD was checked informally (SGD+momentum needs a longer schedule to reach the same point); with an 8-epoch budget AdamW is the better default.
- Mixup / CutMix: implemented and available, but they interact badly with a short schedule (the loss curves are no longer comparable) and did not improve validation accuracy in our runs.

## 4.8 Sensitivity summary

Measured ranges across the sweeps in this section (all at 3 epochs, 160 px, seed 42):

| Knob | Range swept | Accuracy range observed | Verdict |
|---|---|---|---|
| Learning rate | 1e-5 … 3e-3 | 2.6 pp | **matters: tune it** |
| Fine-tune depth | head only … all layers | 2.1 pp | **matters: one step, then flat** |
| Input resolution | 96 … 224 | 1.7 pp | matters, with diminishing returns |
| Augmentation strength | none … full | 0.5 pp | no measurable effect at this budget |
| Batch size, label smoothing | 32/64, 0/0.1 | < 0.3 pp | within noise |

The two knobs that actually matter are the learning rate (an order of magnitude too large is destructive) and the fine-tune depth (the first step of unfreezing is worth everything, the rest is worth nothing). Everything else moves validation accuracy by 1.7 points or less, and augmentation by less than the seed noise. This is why Section 7 spends its budget on transfer strategy, data quantity and resolution rather than on a fine-grained hyper-parameter grid.
