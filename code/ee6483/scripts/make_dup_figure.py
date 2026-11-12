"""Figure: near-duplicates and placeholder graphics found by pHash.

Reads ee6483/results/duplicates_all.json (produced by scripts/find_duplicates.py)
and draws the most informative pairs: a cross-class pair, a cross-split pair, and
the placeholder graphics that appear under both labels.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from ..src.figstyle import apply as _apply_style

_apply_style()
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path("data/datasets")
OUT = Path("ee6483/figures/fig_duplicates.png")

CASES = [
    ("train/cat/cat.8456.jpg", "train/dog/dog.8736.jpg",
     "same image, different labels\n(\"Adopted\" placeholder)"),
    ("train/dog/dog.10401.jpg", "val/cat/cat.11565.jpg",
     "same placeholder graphic\nlabelled dog and cat"),
    ("train/cat/cat.10273.jpg", "val/cat/cat.1639.jpg",
     "true duplicate across the\ntrain/val split"),
]


def main():
    fig, axes = plt.subplots(3, 3, figsize=(6.4, 6.2),
                             gridspec_kw={"width_ratios": [1, 1, 1.25]})
    for r, (a, b, note) in enumerate(CASES):
        for c, rel in enumerate([a, b]):
            ax = axes[r, c]
            ax.imshow(Image.open(ROOT / rel))
            ax.axis("off")
            ax.set_title(Path(rel).name, fontsize=7)
        axes[r, 2].axis("off")
        axes[r, 2].text(0.0, 0.5, note, fontsize=7, va="center", ha="left", style="italic")
    fig.suptitle("pHash near-duplicates (Hamming $\\leq$ 6) in the provided dataset",
                 fontsize=8.5, y=0.99)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print("saved", OUT)

    d = json.loads(Path("ee6483/results/duplicates_all.json").read_text(encoding="utf-8"))
    print(f"pairs={len(d)}  cross_class={sum(1 for x in d if not x['same_class'])}  "
          f"cross_split={sum(1 for x in d if not x['same_split'])}")


if __name__ == "__main__":
    main()
