"""画骨干对比图：(a) 最佳验证准确率 vs 每 epoch 耗时，(b) 4 个代表模型的验证曲线。

python scripts/plot_backbones.py runs/backbones results/backbones.png   # 需要 matplotlib
"""
import csv, statistics as st, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = Path(sys.argv[1]); OUT = Path(sys.argv[2])
models = ["resnet18","resnet34","resnet50","efficientnet_b0","convnext_tiny","swin_t","vit_b_16"]
names = {"resnet18":"ResNet-18","resnet34":"ResNet-34","resnet50":"ResNet-50","efficientnet_b0":"EfficientNet-B0",
         "convnext_tiny":"ConvNeXt-T","swin_t":"Swin-T","vit_b_16":"ViT-B/16"}
H = {}
for m in models:
    H[m] = [list(csv.DictReader(open(R/f"b_{m}_s{s}"/"history.csv"))) for s in (0,1)]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size":10, "axes.edgecolor":INK2, "axes.labelcolor":INK, "xtick.color":INK2, "ytick.color":INK2,
                     "axes.spines.top":False, "axes.spines.right":False, "figure.facecolor":SURF, "axes.facecolor":SURF})
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
# (a) accuracy vs steady-state time per epoch
rows = []
for m in models:
    best = [100*max(float(r["val_accuracy"]) for r in h) for h in H[m]]
    sec = st.median(float(r["seconds"]) for h in H[m] for r in h[1:])
    rows.append((m, st.mean(best), st.stdev(best), sec))
for m, mu, sd, sec in rows:
    a.errorbar(sec, mu, yerr=sd, fmt="o", ms=8, color="#2a78d6", mec=SURF, mew=2, ecolor="#2a78d6", elinewidth=2, capsize=0)
    dx, dy, ha = 1.2, 0.0, "left"
    if m == "resnet34": dy = 0.025
    if m == "resnet18": dy = -0.02; dx = -1.2; ha = "right"
    if m == "swin_t": dy = 0.0
    if m == "resnet50": dy = 0.01; dx=-1.2; ha="right"
    a.annotate(names[m], (sec+dx, mu+dy), va="center", ha=ha, color=INK, fontsize=9)
a.set_xlabel("Training time per epoch on one A40 (s, median of epochs 2–6)")
a.set_ylabel("Best validation accuracy (%)")
a.set_title("(a) Accuracy vs. cost", loc="left", color=INK)
a.grid(True, color=GRID, lw=0.8); a.set_axisbelow(True); a.set_xlim(18, 72)
# (b) validation curves, 4 representative models
pal = {"convnext_tiny":"#2a78d6","resnet50":"#eb6834","vit_b_16":"#1baf7a","resnet18":"#eda100"}
ep = range(1, 7)
for m, c in pal.items():
    mean = [100*st.mean(float(h[i]["val_accuracy"]) for h in H[m]) for i in range(6)]
    b.plot(ep, mean, color=c, lw=2, marker="o", ms=8, mec=SURF, mew=2, label=names[m])
    b.annotate(names[m], (6.12, mean[-1] + {"resnet50":-0.02, "convnext_tiny":0.03, "vit_b_16":0.02, "resnet18":-0.025}[m]),
               va="center", color=INK, fontsize=9)
b.set_xlabel("Epoch"); b.set_ylabel("Validation accuracy (%), mean of 2 seeds")
b.set_title("(b) Validation curves", loc="left", color=INK)
b.set_xlim(0.8, 7.2); b.set_xticks(list(ep)); b.grid(True, color=GRID, lw=0.8); b.set_axisbelow(True)
b.legend(frameon=False, loc="lower right", fontsize=9)
fig.tight_layout(); fig.savefig(OUT, dpi=200)
for r in rows: print("%-16s %.2f ± %.2f  %.1fs" % r)
