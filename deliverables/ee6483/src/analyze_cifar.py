"""Figures for Part G (CIFAR-10) and Part H (class imbalance)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .train_cifar import CIFAR_CLASSES

plt.rcParams.update({"figure.dpi": 130, "font.size": 8.5, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})

METHOD_LABEL = {
    "ce": "CE (baseline)", "resample": "Re-sample\n(balanced)", "cb": "Class-balanced\nloss",
    "ldam": "LDAM-DRW", "decouple": "Decoupling\n(cRT)", "logit_adjust": "Logit\nadjustment",
}


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def save(fig, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)


def fig_cifar_confusion(res: Path, out: Path, exp="G1_cifar_scratch"):
    f = res / exp / "s42" / "metrics.json"
    if not f.exists():
        print("missing", f)
        return
    d = load(f)
    cm = np.array(d["test"]["confusion_matrix"], dtype=float)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(10):
        for j in range(10):
            if cmn[i, j] > 0.01:
                ax.text(j, i, f"{cmn[i,j]*100:.0f}", ha="center", va="center", fontsize=6,
                        color="white" if cmn[i, j] > 0.5 else "black")
    ax.set(xticks=range(10), yticks=range(10), xticklabels=CIFAR_CLASSES, yticklabels=CIFAR_CLASSES,
           xlabel="predicted", ylabel="true",
           title=f"CIFAR-10 confusion matrix (row-normalised %)\ntest accuracy {d['test']['acc']*100:.2f}%")
    ax.grid(False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    save(fig, out / "fig_cifar_confusion.png")


def fig_cifar_perclass(res: Path, out: Path):
    exps = [("G1_cifar_scratch", "ResNet-18-CIFAR from scratch"),
            ("G2_cifar_pretrained", "ImageNet ResNet-18 @160px")]
    data = []
    for e, lab in exps:
        f = res / e / "s42" / "metrics.json"
        if f.exists():
            d = load(f)
            data.append((lab, np.array(d["test"]["per_class_recall"]) * 100, d["test"]["acc"] * 100))
    if not data:
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(10)
    w = 0.8 / len(data)
    for k, (lab, acc, total) in enumerate(data):
        ax.bar(x + k * w - 0.4 + w / 2, acc, w, label=f"{lab}  ({total:.1f}% overall)")
    ax.set(xticks=x, xticklabels=CIFAR_CLASSES, ylabel="per-class accuracy (%)", ylim=[60, 101],
           title="CIFAR-10 per-class accuracy: scratch vs ImageNet-pretrained")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, out / "fig_cifar_perclass.png")


def fig_imbalance(res: Path, out: Path):
    files = sorted(res.glob("H_*/s42/metrics.json"))
    if not files:
        return
    # prefer the fair (longer-schedule) class-balanced run over the short one
    if (res / "H_cb_long" / "s42" / "metrics.json").exists():
        files = [f for f in files if f.parent.parent.name != "H_cb"]
    rows = [(load(f), f) for f in files]
    rows.sort(key=lambda r: r[0]["test"]["acc"])

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    labels = [METHOD_LABEL.get(d["method"], d["method"]) for d, _ in rows]
    acc = [d["test"]["acc"] * 100 for d, _ in rows]
    mf1 = [d["test"]["macro_f1"] * 100 for d, _ in rows]
    x = np.arange(len(rows))
    axes[0].bar(x - 0.2, acc, 0.4, label="overall acc")
    axes[0].bar(x + 0.2, mf1, 0.4, label="macro F1")
    for i, (a, m) in enumerate(zip(acc, mf1)):
        axes[0].text(i - 0.2, a + 0.4, f"{a:.1f}", ha="center", fontsize=6.5)
        axes[0].text(i + 0.2, m + 0.4, f"{m:.1f}", ha="center", fontsize=6.5)
    axes[0].set(xticks=x, xticklabels=labels, ylabel="%", title="CIFAR-10-LT (rho=100): overall vs balanced metric")
    axes[0].set_ylim([min(mf1) - 8, 100])
    axes[0].legend(fontsize=7)
    plt.setp(axes[0].get_xticklabels(), rotation=25, ha="right", fontsize=7)

    # per-class recall of the worst and best method
    worst, best = rows[0][0], rows[-1][0]
    xc = np.arange(10)
    axes[1].plot(xc, np.array(worst["test"]["per_class_recall"]) * 100, "o-", label=f"{worst['method']} (worst)")
    axes[1].plot(xc, np.array(best["test"]["per_class_recall"]) * 100, "s-", label=f"{best['method']} (best)")
    counts = [worst["class_counts"][str(c)] for c in range(10)]
    for c, n in zip(xc, counts):
        axes[1].annotate(f"n={n}", (c, 2), fontsize=5.5, ha="center", color="grey")
    axes[1].set(xticks=xc, xticklabels=CIFAR_CLASSES, ylabel="per-class recall (%)", ylim=[0, 102],
                title="Per-class recall (classes sorted from many to few samples)")
    plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    save(fig, out / "fig_imbalance.png")


def fig_imbalance_curves(res: Path, out: Path):
    files = sorted(res.glob("H_*/s42/metrics.json"))
    if not files:
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for f in files:
        d = load(f)
        h = d["history"]
        ax.plot([x["epoch"] for x in h], [x["val_acc"] * 100 for x in h], lw=1.4,
                label=METHOD_LABEL.get(d["method"], d["method"]).replace("\n", " "))
    ax.set(xlabel="epoch", ylabel="validation accuracy (%)", title="CIFAR-10-LT training curves")
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    save(fig, out / "fig_imbalance_curves.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="ee6483/results")
    ap.add_argument("--out", default="ee6483/figures")
    args = ap.parse_args()
    res, out = Path(args.res), Path(args.out)
    fig_cifar_confusion(res, out)
    fig_cifar_perclass(res, out)
    fig_imbalance(res, out)
    fig_imbalance_curves(res, out)


if __name__ == "__main__":
    main()
