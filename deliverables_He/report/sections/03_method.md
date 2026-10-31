# 3. Model and Training Procedure

## 3.1 Overview of the model ladder

Six models are trained, so that every claim in Section 7 can be measured against a control. All models share the same training engine, the same data pipeline and the same evaluation code; only the architecture and the transfer strategy change.

| ID | Model | Pretrained | Input | Params | Purpose |
|---|---|---|---|---|---|
| M0 | HOG + colour histogram → StandardScaler → SVM (RBF) | – | 96×96 | – | hand-crafted feature ceiling |
| M1 | SmallCNN (4 conv blocks) | No | 128×128 | 1.17M | what is achievable *without* ImageNet |
| M2 | ResNet-18, frozen backbone + linear head | Yes | 224×224 | 11.18M | value of the pre-trained *representation* alone |
| **M3** | **ResNet-18, LP-FT fine-tuned** | Yes | 224×224 | 11.18M | **main model of the study; the final submission ensembles the top backbones of Section 7.8** |
| M4 | ResNet-34, LP-FT fine-tuned | Yes | 224×224 | 21.29M | backbone scaling |
| M5 | ResNet-50, LP-FT fine-tuned | Yes | 160×160 | 23.51M | backbone scaling under a fixed compute budget |

## 3.2 Architecture of the main model (M3)

M3 is a ResNet-18 [He et al., CVPR 2016] with the 1,000-way ImageNet classifier replaced by a two-way head. The residual block is

$$y = \mathrm{ReLU}\big(\mathrm{BN}(W_2\,\mathrm{ReLU}(\mathrm{BN}(W_1 x))) + x\big)$$

with a 1×1 convolution on the shortcut whenever the spatial resolution or channel count changes.

Tensor shapes through the network (batch size $B$, input 224×224):

| Stage | Operation | Output shape | Notes |
|---|---|---|---|
| input | – | $B\times 3\times 224\times 224$ | ImageNet-normalised |
| conv1 | 7×7, 64, stride 2 | $B\times 64\times 112\times 112$ | |
| pool1 | 3×3 max, stride 2 | $B\times 64\times 56\times 56$ | |
| layer1 | 2× BasicBlock(64) | $B\times 64\times 56\times 56$ | stride 1 |
| layer2 | 2× BasicBlock(128) | $B\times 128\times 28\times 28$ | first block stride 2 |
| layer3 | 2× BasicBlock(256) | $B\times 256\times 14\times 14$ | first block stride 2 |
| layer4 | 2× BasicBlock(512) | $B\times 512\times 7\times 7$ | first block stride 2 |
| pool | global average | $B\times 512$ | removes spatial dims |
| head | Dropout(0.2) → Linear(512→2) | $B\times 2$ | **replaces the 1,000-way fc** |
| output | softmax | $B\times 2$ | $p(\text{cat}), p(\text{dog})$ |

![Architecture of the main model with tensor shapes at every stage](../ee6483/figures/fig_architecture.png)

The classification head is a *fresh* layer: its weights are Kaiming-initialised and its bias is zero, so at the start of training the model outputs a near-uniform distribution and the first epochs are spent learning the head, not damaging the backbone. This is the mechanism behind the LP-FT strategy in Section 3.4.

`torchvision.models.resnet18(weights="IMAGENET1K_V1")` supplies the first 512-dimensional feature extractor; everything from `layer1` to `layer4` is loaded from the ImageNet checkpoint.

### M1: SmallCNN (the from-scratch control)

Four blocks, each `Conv3×3-BN-ReLU → Conv3×3-BN-ReLU → MaxPool2`, with 32/64/128/256 channels, followed by global average pooling, dropout 0.3 and a linear layer. At 128×128 input the spatial size goes 128 → 64 → 32 → 16 → 8, so the receptive field of the last block covers the whole image. 1.17M parameters: deliberately an order of magnitude smaller than ResNet-18, on the assumption that a 20k-image dataset cannot support a large model trained from scratch. Section 5 shows this assumption was too pessimistic (it reaches 95.06%), which is itself a finding: the task is easier than the parameter count suggests.

### M0: classical baseline

Each image is resized to 96×96, converted to grayscale for HOG (9 orientations, 8×8 cells, 2×2 blocks, L2-Hys) and kept in RGB for a 16-bin per-channel colour histogram. The two descriptors are concatenated (1764 + 48 = 1812 dimensions), standardised, and classified by an RBF SVM (C=10). Trained on 8,000 images (4,000 per class) because the feature extraction is CPU-bound.

## 3.3 Loss function

$$\mathcal{L} = -\frac{1}{B}\sum_{i=1}^{B}\sum_{c=1}^{2} \tilde{y}_{i,c}\log p_{i,c},\qquad \tilde{y}_{i,c} = (1-\epsilon)\,y_{i,c} + \frac{\epsilon}{2}$$

i.e. categorical cross-entropy with label smoothing $\epsilon=0.1$. Two reasons:

1. Web-sourced labels contain a small fraction of errors (multi-animal images, mislabels). With hard targets the loss pushes the model to be infinitely confident on those samples; smoothing caps the target logit at $1-\epsilon+\epsilon/2$ and stops the model from fitting label noise.
2. It improves calibration: relevant because Section 6 reports prediction confidences and because the ensemble in Section 5 averages probabilities.

For the CIFAR-10 experiments the same loss is used with $C=10$; the only change is the output dimension, which keeps the two parts of the project directly comparable.

## 3.4 Training strategy: LP-FT

Kumar et al. [ICLR 2022] show that naive fine-tuning distorts pre-trained features: a large gradient step on a randomly initialised head propagates back into the backbone and moves it away from the pre-trained optimum *before* the head has learned to read the features. Their fix, LP-FT (linear-probe then fine-tune), is two-staged:

```
Stage 1 (epochs 1..2):  freeze layer1..layer4,  train head only,  lr_head = 1e-3
Stage 2 (epochs 3..8):  unfreeze all,            lr_backbone = 3e-4, lr_head = 1e-3,
                        cosine decay to 0, 0.3-epoch warm-up
```

This is visible in our training log as a small accuracy dip at the stage-2 transition (97.2% → 95.0% at epoch 3 for seed 42) followed by recovery to 98.4% at epoch 4: direct evidence that the phenomenon described in the paper also occurs here, and that the second stage needs its own warm-up.

Full recipe

| Component | Setting |
|---|---|
| Optimiser | AdamW, β=(0.9, 0.999), weight decay 1e-4 |
| Backbone LR | 3e-4 (stage 2) |
| Head LR | 1e-3 |
| Schedule | linear warm-up then cosine to 0, stepped per iteration |
| Batch size | 64 |
| Epochs | 8 (early stopping, patience 3 on validation accuracy) |
| Precision | mixed precision (fp16 autocast + GradScaler) |
| Gradient clipping | global norm 1.0 |
| Input | 224×224, ImageNet mean/std |

Why each choice (this is the content of Section 4, stated here for completeness):

- *AdamW over SGD*: with 8 epochs and a small dataset, SGD+momentum needs a longer schedule and a careful LR sweep to match AdamW. AdamW's decoupled weight decay also means the regularisation strength does not interact with the adaptive LR.
- *Differential LR*: the head is randomly initialised and needs a large step; the backbone is already at a good optimum and needs a small one. A single LR forces a compromise that is either too slow for the head or destructive for the backbone.
- *Cosine + warm-up*: warm-up avoids the early large steps that destabilise the head; cosine annealing spends most of the budget in the low-LR region where the model settles into a flatter minimum.
- *Batch 64*: 8 GB of VRAM fits 96 at 224 px, but batch 64 keeps BatchNorm statistics stable and gives 312 iterations per epoch, which is enough for a smooth cosine schedule over 8 epochs.

## 3.5 Data pipeline

```
train : RandomResizedCrop(224, scale=(0.5,1.0))
      → RandomHorizontalFlip(p=0.5)
      → RandomRotation(15°)
      → ColorJitter(0.2,0.2,0.2,0.05)
      → RandAugment(num_ops=2, magnitude=7)
      → ToTensor → Normalize(ImageNet) → RandomErasing(p=0.25)
val/test : Resize(256) → CenterCrop(224) → ToTensor → Normalize(ImageNet)
```

Details that matter: no vertical flip (an upside-down animal is out of distribution and the model loses accuracy); `RandomResizedCrop` rather than `Resize+RandomCrop` so that scale jitter and translation jitter are applied jointly; `RandomErasing` specifically to break the "dog on grass / cat on sofa" background shortcut.

## 3.6 Reproducibility

- `set_seed(s)` fixes `PYTHONHASHSEED`, `random`, `numpy`, `torch` and `torch.cuda` for $s \in \{42,43,44\}$, with `cudnn.deterministic=True` and `cudnn.benchmark=False`.
- AMP kernels are not bit-exact even with deterministic flags, so every headline number is reported as mean ± std over three seeds, not as a single run.
- Each run writes `metrics.json` (with the full argument dictionary), `curves.csv`, `train.log` and `best.pt`; the orchestrator appends the exact command line to `results/run_log.txt`.
- Data are cached once at short-side 256 px, so every experiment sees identical pixels regardless of when it ran.
- k-fold cross-validation is not used. The assignment fixes the train/validation split, and re-partitioning it would change the validation set that every reported number refers to, making the results incomparable across models and with the provided baseline. Three seeds serve the same purpose, estimating run-to-run variance, while keeping a single fixed validation set.

## 3.7 Code and how to run it

The complete implementation is submitted with this report as the `ee6483/` package: 30 Python modules, the configuration files, every run's `metrics.json` and training log, and the 24 figures. `README.md` at the repository root lists every file and what it does, and `ee6483/README.md` gives the step-by-step reproduction commands. Minimum working example:

```bash
# main model, one seed
python -m ee6483.src.train --exp-id E4_r18_ft --model resnet18 --epochs 8 \
    --lr 3e-4 --head-lr 1e-3 --batch 64 --img-size 224 --aug full \
    --ft-strategy lpft --freeze-epochs 2 --seed 42 --data-root data/cache

# first submission: 3-seed M3 ensemble with horizontal-flip TTA
python -m ee6483.src.predict --ckpts \
    ee6483/results/E4_r18_ft/s42/best.pt \
    ee6483/results/E4_r18_ft/s43/best.pt \
    ee6483/results/E4_r18_ft/s44/best.pt --out submissions/submission_m3_ensemble.csv --tta 1

# final submission: top-tier backbones of Section 7.8 (cluster; see CLUSTER_GUIDE.md)
sbatch cluster/backbones.sbatch      # 7 backbones x 2 seeds
sbatch cluster/ensemble.sbatch       # -> submissions/submission_backbone_ensemble.csv

# strict format check before uploading
python ee6483/scripts/validate_submission.py --sub submissions/submission.csv
```

| Module | Responsibility |
|---|---|
| `src/train.py` | single-run trainer; every hyper-parameter is a CLI flag |
| `src/engine.py` | train/eval loops, all metrics (acc, macro-F1, per-class P/R, AUC, confusion matrix) |
| `src/data/` | transforms, ImageFolder loaders with class-balanced subsets, resize cache, dataset health check |
| `src/models/` | backbones, SmallCNN, CIFAR-ResNet |
| `src/losses.py` | CE / label smoothing / class-balanced / LDAM / logit adjustment |
| `src/predict.py` | ensemble + TTA → `submission.csv` |
| `src/gradcam.py` | Grad-CAM for Section 6 |
| `scripts/` | experiment orchestrator, result aggregation, submission validator |
