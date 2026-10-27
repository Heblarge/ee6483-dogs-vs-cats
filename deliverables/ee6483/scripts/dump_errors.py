"""Dump every validation error as an image grid, so the failure taxonomy in
Part E is based on looking at all errors rather than on a handful of examples.

    python -m ee6483.scripts.dump_errors --ckpt ... --out ee6483/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from ..src.figstyle import apply as _apply_style

_apply_style()
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from ..src.analyze import Perturb, PerturbedFolder, collect_val_preds, load_model
from ..src.data.transforms import build_eval_transform
from ..src.utils import get_device

NAMES = ["cat", "dog"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ee6483/results/E4_r18_ft/s42/best.pt")
    ap.add_argument("--data-root", default="data/cache")
    ap.add_argument("--out", default="ee6483/figures")
    ap.add_argument("--per-fig", type=int, default=25)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--per-dir", type=int, default=5)
    args = ap.parse_args()

    device = get_device()
    model, a = load_model(args.ckpt, device)
    root = Path(args.data_root)
    ds = PerturbedFolder(root / "val", build_eval_transform(a["img_size"]), Perturb("none"))
    probs, tgts = collect_val_preds(model, DataLoader(ds, batch_size=128, num_workers=4), device)
    preds = probs.argmax(1)
    wrong = np.where(preds != tgts)[0]
    wrong = wrong[np.argsort(-probs[wrong].max(1))]
    print(f"{len(wrong)} errors")

    # A class-balanced panel: the most confident errors in each direction.
    # The raw error counts are asymmetric (32 dog->cat against 18 cat->dog), so a
    # plain top-k by confidence over-represents the cat->dog direction.
    per_dir = args.per_dir
    cat2dog = [i for i in wrong if tgts[i] == 0][:per_dir]
    dog2cat = [i for i in wrong if tgts[i] == 1][:per_dir]
    if cat2dog and dog2cat:
        sel = sorted(cat2dog + dog2cat, key=lambda i: -probs[i].max())
        cols, rows = 5, (len(sel) + 4) // 5
        fig, axes = plt.subplots(rows, cols, figsize=(6.4, 1.95 * rows))
        axes = np.atleast_1d(axes).ravel()
        for k, i in enumerate(sel):
            axes[k].imshow(Image.open(ds.files[i]))
            axes[k].axis("off")
            axes[k].set_title(f"true={NAMES[tgts[i]]} pred={NAMES[preds[i]]}\np={probs[i].max():.2f}",
                              fontsize=6.5)
        for k in range(len(sel), len(axes)):
            axes[k].axis("off")
        fig.tight_layout()
        out = Path(args.out) / "fig_errors_balanced.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("saved", out)

    cols = args.cols
    for fig_i in range(0, len(wrong), args.per_fig):
        chunk = wrong[fig_i: fig_i + args.per_fig]
        rows = (len(chunk) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6.4, 1.95 * rows))
        axes = np.atleast_1d(axes).ravel()
        for k, i in enumerate(chunk):
            axes[k].imshow(Image.open(ds.files[i]))
            axes[k].axis("off")
            axes[k].set_title(f"#{fig_i+k+1} {Path(ds.files[i]).name}\n"
                              f"true={NAMES[tgts[i]]} pred={NAMES[preds[i]]} p={probs[i].max():.2f}",
                              fontsize=6.5)
        for k in range(len(chunk), len(axes)):
            axes[k].axis("off")
        fig.tight_layout()
        out = Path(args.out) / f"fig_errors_all_{fig_i//args.per_fig+1}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("saved", out)


if __name__ == "__main__":
    main()
