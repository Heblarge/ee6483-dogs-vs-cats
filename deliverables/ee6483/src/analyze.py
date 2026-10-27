"""Figures for the Dogs vs Cats report (Parts A, D, E, F).

All figures land in ee6483/figures/. Every function degrades gracefully if the
corresponding result folder does not exist yet, so this can be re-run at any
point during the project.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from .figstyle import apply as _apply_style

_apply_style()
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset

from .data.transforms import build_eval_transform
from .models.backbones import build_backbone
from .models.small_cnn import SmallCNN
from .utils import get_device

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})
NAMES = ["cat", "dog"]


# ------------------------------------------------------------------ helpers
def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("saved", path)


def load_model(ckpt, device):
    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    a = d["args"]
    m = SmallCNN() if a["model"] == "smallcnn" else build_backbone(a["model"], 2, pretrained=False)
    m.load_state_dict(d["model"])
    return m.to(device).eval(), a


# ------------------------------------------------------------------ dataset figs
def _square(im, size: int = 320):
    """Centre-crop to a square thumbnail so an image grid lines up."""
    w, h = im.size
    side = min(w, h)
    im = im.crop(((w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2))
    return im.resize((size, size), Image.BICUBIC)


def fig_dataset_samples(root: Path, out: Path, n: int = 4):
    fig, axes = plt.subplots(2, n, figsize=(6.4, 3.6))
    for r, cls in enumerate(["cat", "dog"]):
        files = sorted((root / "train" / cls).glob("*.jpg"))[: n * 37 : 37][:n]
        for c, p in enumerate(files):
            axes[r, c].imshow(_square(Image.open(p).convert("RGB")))
            axes[r, c].set_box_aspect(1)
            axes[r, c].axis("off")
            if c == 0:
                axes[r, c].set_ylabel(cls, fontsize=11)
    fig.suptitle("Training samples: large intra-class variation in breed, pose, scale and background",
                 fontsize=8.5, y=1.02)
    fig.tight_layout()
    save(fig, out / "fig_dataset_samples.png")


def fig_augmentation(root: Path, out: Path, img_size: int = 224, n: int = 6):
    from .data.transforms import build_train_transform
    from .data.transforms import IMAGENET_MEAN, IMAGENET_STD

    tf = build_train_transform(img_size, "full")
    p = sorted((root / "train" / "dog").glob("*.jpg"))[100]
    img = Image.open(p).convert("RGB")
    ncol = 4
    nrow = (n + 1 + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.4, 1.75 * nrow))
    axes = np.atleast_1d(axes).ravel()
    axes[0].imshow(img)
    axes[0].set_title("original", fontsize=7.5)
    axes[0].axis("off")
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    for i in range(1, n + 1):
        t = tf(img)
        im = (t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        axes[i].imshow(im)
        axes[i].set_title(f"aug #{i}", fontsize=7.5)
        axes[i].axis("off")
    for k in range(n + 1, len(axes)):
        axes[k].axis("off")
    fig.suptitle("RandomResizedCrop + flip + rotation + ColorJitter + RandAugment + RandomErasing",
                 fontsize=8.5, y=1.02)
    fig.tight_layout()
    save(fig, out / "fig_augmentation.png")


# ------------------------------------------------------------------ training curves
def fig_curves(res_root: Path, out: Path, exps: list[str]):
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.6))
    for exp in exps:
        f = res_root / exp / "s42" / "curves.csv"
        if not f.exists():
            continue
        rows = np.genfromtxt(f, delimiter=",", names=True)
        axes[0].plot(rows["epoch"], rows["train_loss"], marker="o", ms=3, label=f"{exp} train")
        axes[0].plot(rows["epoch"], rows["val_loss"], marker="s", ms=3, ls="--", label=f"{exp} val")
        axes[1].plot(rows["epoch"], rows["val_acc"] * 100, marker="o", ms=3, label=exp)
    axes[0].set(xlabel="epoch", ylabel="cross-entropy loss", title="Loss")
    axes[1].set(xlabel="epoch", ylabel="validation accuracy (%)", title="Validation accuracy")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    save(fig, out / "fig_training_curves.png")


# ------------------------------------------------------------------ confusion / errors
@torch.no_grad()
def collect_val_preds(model, loader, device):
    probs, tgts = [], []
    for imgs, t in loader:
        imgs = imgs.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(imgs).float()
        probs.append(F.softmax(logits, 1).cpu().numpy())
        tgts.append(t.numpy())
    return np.concatenate(probs), np.concatenate(tgts)


def fig_confusion(probs, tgts, out: Path, tag="dogs"):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(tgts, probs.argmax(1), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]}\n({cm[i,j]/cm[i].sum()*100:.2f}%)", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=10)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=NAMES, yticklabels=NAMES,
           xlabel="predicted", ylabel="true", title=f"Confusion matrix ({tag})")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    save(fig, out / f"fig_confusion_{tag}.png")


def fig_errors(probs, tgts, root: Path, out: Path, n: int = 8, conf: bool = True):
    preds = probs.argmax(1)
    wrong = np.where(preds != tgts)[0]
    order = wrong[np.argsort(-probs[wrong].max(1))][:n] if len(wrong) else wrong
    # ImageFolder order: cat first, then dog, each sorted by name
    files = sorted((root / "val" / "cat").glob("*.jpg")) + sorted((root / "val" / "dog").glob("*.jpg"))
    if not len(order):
        print("no misclassified validation samples found")
        return
    cols = 4
    rows = (len(order) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6.4, 1.9 * rows))
    axes = np.atleast_1d(axes).ravel()
    for k, i in enumerate(order):
        axes[k].imshow(Image.open(files[i]))
        axes[k].axis("off")
        t = f"true={NAMES[tgts[i]]}  pred={NAMES[preds[i]]}\np={probs[i].max():.3f}" if conf else \
            f"true={NAMES[tgts[i]]}  pred={NAMES[preds[i]]}"
        axes[k].set_title(t, fontsize=7)
    for k in range(len(order), len(axes)):
        axes[k].axis("off")
    fig.suptitle("Highest-confidence misclassifications on the validation set", fontsize=8.5, y=1.02)
    fig.tight_layout()
    save(fig, out / "fig_error_cases.png")
    return order


# ------------------------------------------------------------------ robustness
class Perturb:
    """Picklable perturbation (Windows DataLoader workers cannot pickle lambdas)."""

    def __init__(self, name: str):
        self.name = name

    def __call__(self, im):
        n = self.name
        if n == "none":
            return im
        if n == "blur2":
            return im.filter(ImageFilter.GaussianBlur(2))
        if n == "blur4":
            return im.filter(ImageFilter.GaussianBlur(4))
        if n == "bright":
            return Image.fromarray(np.clip(np.asarray(im, np.float32) * 1.6, 0, 255).astype(np.uint8))
        if n == "dark":
            return Image.fromarray(np.clip(np.asarray(im, np.float32) * 0.45, 0, 255).astype(np.uint8))
        if n == "rot30":
            return im.rotate(30, expand=False)
        if n == "occ25":
            a = np.asarray(im).copy()
            h, w = a.shape[:2]
            s = int(0.25 * min(h, w))
            cy, cx = h // 2, w // 2
            a[cy - s // 2: cy + s // 2, cx - s // 2: cx + s // 2] = 128
            return Image.fromarray(a)
        if n == "noise":
            rng = np.random.RandomState(0)
            a = np.asarray(im, np.float32)
            return Image.fromarray(np.clip(a + rng.normal(0, 25, a.shape), 0, 255).astype(np.uint8))
        raise ValueError(n)


class PerturbedFolder(Dataset):
    def __init__(self, root: Path, tf, fn):
        cats = sorted((root / "cat").glob("*.jpg"))
        dogs = sorted((root / "dog").glob("*.jpg"))
        self.files = cats + dogs
        self.labels = [0] * len(cats) + [1] * len(dogs)
        self.tf, self.fn = tf, fn

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        im = Image.open(self.files[i]).convert("RGB")
        return self.tf(self.fn(im)), self.labels[i]


def fig_robustness(model, root: Path, img_size: int, out: Path, device):
    tf = build_eval_transform(img_size)
    names = ["none", "blur2", "blur4", "bright", "dark", "rot30", "occ25", "noise"]
    accs = []
    for nm in names:
        ds = PerturbedFolder(root / "val", tf, Perturb(nm))
        dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=4)
        p, t = collect_val_preds(model, dl, device)
        accs.append((p.argmax(1) == t).mean() * 100)
        print(f"  robustness {nm}: {accs[-1]:.2f}%")
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    bars = ax.bar(names, accs, color=["#4c72b0"] + ["#dd8452"] * (len(names) - 1))
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.3, f"{a:.1f}", ha="center", fontsize=8)
    ax.set(ylabel="validation accuracy (%)", title="Robustness to perturbations (clean model, no retraining)",
           ylim=[min(accs) - 5, 100])
    ax.axhline(accs[0], ls="--", c="grey", lw=1)
    fig.tight_layout()
    save(fig, out / "fig_robustness.png")
    return dict(zip(names, [round(a, 2) for a in accs]))


# ------------------------------------------------------------------ ablation plots
def fig_ablation(res_root: Path, out: Path):
    """Six panels, one per factor, with a fixed logical ordering.

    The 3-epoch reference run (E7_aug_full: 160 px, full augmentation) is reused
    as the "160 px", "all data" and "full fine-tune" level of the other panels,
    and is drawn in a different colour so the reference is unambiguous.
    """
    import json as _json

    def acc(exp):
        f = res_root / exp / "s42" / "metrics.json"
        if not f.exists():
            return None
        return _json.loads(f.read_text(encoding="utf-8"))["best_val_acc"] * 100

    REF = "E7_aug_full"
    panels = [
        ("augmentation", [("none", "E7_aug_none"), ("flip", "E7_aug_flip"), ("+geo", "E7_aug_geo"),
                          ("+color", "E7_aug_color"), ("+randaug\n+erasing", REF)]),
        ("resolution", [("96", "E7_res_96"), ("128", "E7_res_128"), ("160", REF), ("224", "E7_res_224")]),
        ("training data", [("250", "E7_data_250"), ("1k", "E7_data_1000"),
                           ("2.5k", "E7_data_2500"), ("5k", "E7_data_5000"), ("10k", REF)]),
        ("fine-tune depth", [("head\nonly", "E7_ft_head"), ("last\nstage", "E7_ft_last"),
                             ("all\n(naive)", REF), ("all\n(LP-FT)", "E7_ft_lpft")]),
        ("learning rate", [("1e-5", "E7_lr_1e-5"), ("1e-4", "E7_lr_1e-4"), ("3e-4", "E7_lr_3e-4"),
                           ("1e-3", "E7_lr_1e-3"), ("3e-3", "E7_lr_3e-3")]),
        ("pre-training", [("random\ninit", "E7_scratch"), ("ImageNet", REF)]),
    ]
    panels = [(t, [(lab, e, acc(e)) for lab, e in items if acc(e) is not None]) for t, items in panels]
    panels = [(t, items) for t, items in panels if items]
    if not panels:
        return

    ncol = 3
    nrow = (len(panels) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.5, 2.45 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, items) in zip(axes, panels):
        labels = [i[0] for i in items]
        vals = [i[2] for i in items]
        colors = ["#c44e52" if i[1] == REF else "#4c72b0" for i in items]
        bars = ax.bar(labels, vals, color=colors)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.12, f"{v:.1f}", ha="center", fontsize=6.0)
        lo, hi = min(vals), max(vals)
        pad = max(1.2, (hi - lo) * 0.45)
        ax.set_ylim([lo - pad, hi + pad * 0.75])
        ax.set_title(title, fontsize=8)
        if title == "training data":
            ax.set_xlabel("images per class", fontsize=6.5)
        ax.tick_params(axis="x", labelsize=6.0)
        ax.tick_params(axis="y", labelsize=6.0)
        ax.set_ylabel("val acc (%)", fontsize=6.5)
    fig.suptitle("One-factor-at-a-time ablation, ResNet-18 @160px, 3 epochs, seed 42 "
                 "(red = shared reference)", y=1.01, fontsize=8)
    fig.tight_layout()
    save(fig, out / "fig_ablation.png")


def fig_test_confidence(root: Path, probs_path: Path, out: Path, n_low: int = 8, n_high: int = 4):
    """Test-set predictions: least and most confident, balanced across predicted classes.

    The test set is unlabelled, so this is a qualitative check only. Two
    corrections keep the panel representative: the picks are balanced between
    the two predicted classes, and near-duplicate images (the dataset contains
    repeated placeholder graphics) are suppressed.
    """
    if not probs_path.exists():
        print("no submission probs found, skip test figure")
        return
    import imagehash

    probs = np.load(probs_path)
    test_files = sorted((root / "test").glob("*.jpg"), key=lambda p: int(p.stem))
    ids = np.array([int(p.stem) for p in test_files])
    pred = probs.argmax(1)
    conf = probs.max(1)

    # perceptual hashes, so the same placeholder graphic is not shown twice
    hashes = {}
    for i, f in enumerate(test_files):
        with Image.open(f) as im:
            hashes[i] = imagehash.phash(im.convert("L").resize((64, 64)))

    def pick(order):
        chosen = []
        for i in order:
            if len(chosen) >= len(pred) // 2:
                break
            if any(hashes[i] - hashes[j] <= 4 for j in chosen):
                continue
            if sum(1 for j in chosen if pred[j] == pred[i]) >= max(1, len(order) // 4):
                continue
            chosen.append(i)
        return chosen

    low, high = [], []
    for cls in (0, 1):
        idx = np.where(pred == cls)[0]
        idx = idx[np.argsort(conf[idx])]
        low += list(idx[: n_low // 2])
        high += list(idx[-n_high // 2:][::-1])
    picks = low + high

    ncol = 4
    nrow = (len(picks) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.4, 1.85 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, i in zip(axes, picks):
        ax.imshow(Image.open(test_files[i]))
        ax.axis("off")
        ax.set_title(f"id={ids[i]}  {NAMES[pred[i]]} {conf[i]:.2f}", fontsize=6.8)
    for ax in axes[len(picks):]:
        ax.axis("off")
    fig.suptitle("Test set (unlabelled): four least-confident predictions per class (top two rows) "
                 "and two most-confident per class (bottom row)", fontsize=8.5, y=1.01)
    fig.tight_layout()
    save(fig, out / "fig_test_confidence.png")


def fig_learning_curve(res_root: Path, out: Path):
    import csv

    f = res_root / "summary.csv"
    if not f.exists():
        return
    rows = list(csv.DictReader(open(f, encoding="utf-8")))
    # only runs from the same ablation budget (3 epochs, 160 px) -- mixing in the
    # 8-epoch 224 px main model would make the curve incomparable
    pts = [(int(r["n_train"]), float(r["val_acc"]), r["exp"]) for r in rows
           if r["exp"].startswith("E7_data") or r["exp"] == "E7_aug_full"]
    if len(pts) < 2:
        return
    pts.sort()
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color="#4c72b0")
    for n, a, e in pts:
        ax.annotate(f"{a:.1f}", (n, a), textcoords="offset points", xytext=(0, 6), fontsize=7, ha="center")
    ax.set(xscale="log", xlabel="number of training images (log scale)",
           ylabel="validation accuracy (%)", title="Effect of training-set size (ResNet-18, pretrained)")
    fig.tight_layout()
    save(fig, out / "fig_learning_curve.png")


def fig_model_comparison(res_root: Path, out: Path):
    """Bar chart of the model ladder, with mean/std over seeds where available."""
    import csv

    want = [("E1_hog_svm", "HOG+SVM\n(hand-crafted)"), ("E2_smallcnn", "SmallCNN\n(scratch)"),
            ("E3_r18_frozen", "ResNet18\nfrozen"), ("E4_r18_ft", "ResNet18\nfine-tuned"),
            ("E5_r34_ft", "ResNet34\nfine-tuned"), ("E6_r50_ft", "ResNet50\nfine-tuned")]
    vals = []
    for exp, lab in want:
        if exp == "E1_hog_svm":
            f = res_root / "E1_hog_svm" / "metrics.json"
            if f.exists():
                d = json.loads(f.read_text(encoding="utf-8"))
                vals.append((lab, d["svm_rbf"]["val_acc"] * 100, 0.0, 1))
            continue
        rs = []
        for mf in sorted((res_root / exp).glob("s*/metrics.json")):
            rs.append(json.loads(mf.read_text(encoding="utf-8"))["best_val_acc"] * 100)
        if rs:
            vals.append((lab, float(np.mean(rs)), float(np.std(rs)), len(rs)))
    if not vals:
        return
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    colors = ["#8c8c8c", "#dd8452", "#55a868", "#4c72b0", "#4c72b0", "#4c72b0"]
    bars = ax.bar([v[0] for v in vals], [v[1] for v in vals],
                  yerr=[v[2] for v in vals], capsize=3, color=colors[:len(vals)])
    for b, v in zip(bars, vals):
        inside = v[1] > 80
        ax.text(b.get_x() + b.get_width() / 2, v[1] - 4.5 if inside else v[1] + 1.2, f"{v[1]:.2f}",
                ha="center", fontsize=8, color="white" if inside else "black", fontweight="bold")
        if inside:
            ax.text(b.get_x() + b.get_width() / 2, 71.5, f"n={v[3]}", ha="center", fontsize=6.5, color="white")
    ax.set(ylabel="validation accuracy (%)", ylim=[70, 101],
           title="Model ladder: hand-crafted features, frozen backbone, fine-tuned CNNs")
    fig.tight_layout()
    save(fig, out / "fig_model_comparison.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/cache")
    ap.add_argument("--res-root", default="ee6483/results")
    ap.add_argument("--out", default="ee6483/figures")
    ap.add_argument("--ckpt", default="ee6483/results/E4_r18_ft/s42/best.pt")
    ap.add_argument("--which", default="all")
    args = ap.parse_args()

    root, res_root, out = Path(args.data_root), Path(args.res_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device()

    if args.which in ("all", "data"):
        fig_dataset_samples(root, out)
        fig_augmentation(root, out)
    if args.which in ("all", "curves"):
        fig_curves(res_root, out, ["E4_r18_ft", "E2_smallcnn", "E3_r18_frozen"])
    if args.which in ("all", "ablation"):
        fig_ablation(res_root, out)
        fig_learning_curve(res_root, out)
        fig_model_comparison(res_root, out)
    if args.which in ("all", "error"):
        ck = Path(args.ckpt)
        if not ck.exists():
            print("checkpoint missing, skip error analysis")
            return
        model, a = load_model(ck, device)
        ds = PerturbedFolder(root / "val", build_eval_transform(a["img_size"]), Perturb("none"))
        probs, tgts = collect_val_preds(model, DataLoader(ds, batch_size=128, num_workers=4), device)
        fig_confusion(probs, tgts, out)
        fig_errors(probs, tgts, root, out)
        rob = fig_robustness(model, root, a["img_size"], out, device)
        (out / "robustness.json").write_text(json.dumps(rob, indent=2), encoding="utf-8")
        fig_test_confidence(root, Path("submission.probs.npy"), out)
        # summary numbers for the report
        acc = (probs.argmax(1) == tgts).mean()
        print(f"val accuracy recomputed: {acc*100:.2f}%  errors={int((probs.argmax(1)!=tgts).sum())}")


if __name__ == "__main__":
    main()
