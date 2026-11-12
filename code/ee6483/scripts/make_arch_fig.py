"""Draw the model architecture figure (Part B requires a figure with dimensions)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

STAGES = [
    ("input", "3×224×224", "#dfe6ee"),
    ("conv1 7×7/2\n64, BN, ReLU", "64×112×112", "#c9d8e8"),
    ("maxpool\n3×3/2", "64×56×56", "#c9d8e8"),
    ("layer1\n2×BB(64)", "64×56×56", "#a8c4dd"),
    ("layer2\n2×BB(128)/2", "128×28×28", "#a8c4dd"),
    ("layer3\n2×BB(256)/2", "256×14×14", "#a8c4dd"),
    ("layer4\n2×BB(512)/2", "512×7×7", "#a8c4dd"),
    ("global\navg pool", "512", "#f2d6a2"),
    ("dropout 0.2\nFC 512→2", "2", "#e8b86d"),
    ("softmax", "p(cat), p(dog)", "#d99b4a"),
]


def main(out="ee6483/figures/fig_architecture.png"):
    """Two rows of five stages so the figure fits a 16 cm column at ~1:1 scale."""
    fig, ax = plt.subplots(figsize=(6.5, 3.1))
    w, h, gap = 1.0, 0.72, 0.26
    per_row = 5
    for i, (name, shape, color) in enumerate(STAGES):
        row, col = divmod(i, per_row)
        x = col * (w + gap)
        y = 1.55 if row == 0 else 0.25
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
                                             boxstyle="round,pad=0.02,rounding_size=0.06",
                                             fc=color, ec="#333333", lw=0.8))
        ax.text(x + w / 2, y + h * 0.66, name, ha="center", va="center", fontsize=6.2)
        ax.text(x + w / 2, y + h * 0.22, shape, ha="center", va="center", fontsize=6.0,
                style="italic", color="#1a1a1a")
        if col < per_row - 1 and i < len(STAGES) - 1:
            ax.annotate("", xy=(x + w + gap, y + h / 2), xytext=(x + w, y + h / 2),
                        arrowprops=dict(arrowstyle="->", lw=0.8, color="#333333"))
        ax.text(x + 0.07, y + h - 0.09, str(i + 1), ha="left", va="top",
                fontsize=6.0, color="#666666")
    ax.set_xlim(-0.15, per_row * (w + gap))
    ax.set_ylim(0.05, 2.42)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
