"""Sequential experiment orchestrator (one GPU, so no parallelism).

    python -m ee6483.scripts.run_experiments --phases main ablation

Each job is a subprocess so a crash in one run does not kill the queue, and
every job appends its command line to results/run_log.txt for reproducibility.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
LOG = Path("ee6483/results/run_log.txt")
DATA = "data/cache"


def job(cmd: list[str], tag: str):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"\n### {time.strftime('%Y-%m-%d %H:%M:%S')} {tag}\n{' '.join(cmd)}\n")
    print(f"\n=== {tag} ===", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    dt = time.time() - t0
    tail = (r.stdout or "").strip().splitlines()[-6:]
    with open(LOG, "a", encoding="utf-8") as f:
        f.write("\n".join(tail) + f"\n(rc={r.returncode}, {dt:.0f}s)\n")
        if r.returncode != 0:
            f.write("STDERR:\n" + (r.stderr or "")[-3000:] + "\n")
    print("\n".join(tail), flush=True)
    print(f"[{tag}] rc={r.returncode} {dt:.0f}s", flush=True)
    return r.returncode


def train(exp, model, epochs, lr=3e-4, head_lr=1e-3, batch=64, img=224, aug="full",
          ft="lpft", freeze=2, seed=42, n_per_class=0, pretrained=1, patience=3, extra=None):
    cmd = [PY, "-m", "ee6483.src.train", "--exp-id", exp, "--model", model,
           "--epochs", str(epochs), "--lr", str(lr), "--head-lr", str(head_lr),
           "--batch", str(batch), "--img-size", str(img), "--aug", aug,
           "--ft-strategy", ft, "--freeze-epochs", str(freeze), "--seed", str(seed),
           "--n-per-class", str(n_per_class), "--pretrained", str(pretrained),
           "--data-root", DATA, "--workers", "4", "--patience", str(patience)]
    return cmd + (extra or [])


PHASES = {
    "main": [
        (train("E4_r18_ft", "resnet18", 8, seed=s), f"E4 resnet18 lpft seed{s}") for s in (42, 43, 44)
    ],
    "compare": [
        ([PY, "-m", "ee6483.src.classical"], "E1 HOG+SVM"),
        (train("E2_smallcnn", "smallcnn", 14, lr=1e-3, head_lr=1e-3, img=128, ft="scratch", patience=6), "E2 smallcnn scratch"),
        (train("E3_r18_frozen", "resnet18", 6, lr=1e-3, head_lr=1e-3, ft="head", patience=6), "E3 resnet18 frozen head"),
    ] + [(train("E5_r34_ft", "resnet34", 8, seed=s), f"E5 resnet34 lpft seed{s}") for s in (42,)],
    "ablation": [
        # one factor at a time, ResNet-18 @160, 3 epochs, seed 42
        (train("E7_aug_none", "resnet18", 3, img=160, aug="none", ft="full"), "aug none"),
        (train("E7_aug_flip", "resnet18", 3, img=160, aug="flip", ft="full"), "aug flip"),
        (train("E7_aug_geo", "resnet18", 3, img=160, aug="geo", ft="full"), "aug geo"),
        (train("E7_aug_color", "resnet18", 3, img=160, aug="color", ft="full"), "aug color"),
        (train("E7_aug_full", "resnet18", 3, img=160, aug="full", ft="full"), "aug full (base)"),
        (train("E7_res_96", "resnet18", 3, img=96, aug="full", ft="full"), "res 96"),
        (train("E7_res_128", "resnet18", 3, img=128, aug="full", ft="full"), "res 128"),
        (train("E7_res_224", "resnet18", 3, img=224, aug="full", ft="full"), "res 224"),
        (train("E7_data_250", "resnet18", 3, img=160, aug="full", ft="full", n_per_class=250), "data 250/class"),
        (train("E7_data_1000", "resnet18", 3, img=160, aug="full", ft="full", n_per_class=1000), "data 1000/class"),
        (train("E7_data_2500", "resnet18", 3, img=160, aug="full", ft="full", n_per_class=2500), "data 2500/class"),
        (train("E7_data_5000", "resnet18", 3, img=160, aug="full", ft="full", n_per_class=5000), "data 5000/class"),
        (train("E7_ft_head", "resnet18", 3, img=160, aug="full", ft="head"), "ft head only"),
        (train("E7_ft_last", "resnet18", 3, img=160, aug="full", ft="last_stage"), "ft last stage"),
        (train("E7_ft_lpft", "resnet18", 3, img=160, aug="full", ft="lpft", freeze=1), "ft lp-ft"),
        (train("E7_scratch", "resnet18", 6, img=160, aug="full", ft="scratch", patience=6), "pretrained vs scratch"),
    ],
    "cifar": [
        ([PY, "-m", "ee6483.src.train_cifar", "--mode", "g1", "--epochs", "25", "--exp-id", "G1_cifar_scratch"], "G1 cifar scratch"),
        ([PY, "-m", "ee6483.src.train_cifar", "--mode", "g2", "--epochs", "7", "--exp-id", "G2_cifar_pretrained"], "G2 cifar pretrained"),
    ],
    "imbalance": [
        ([PY, "-m", "ee6483.src.train_cifar", "--mode", "h", "--method", m, "--epochs", "18",
          "--exp-id", f"H_{m}"], f"H {m}") for m in ("ce", "resample", "cb", "decouple", "logit_adjust")
    ],
    "extra": [
        (train("E6_r50_ft", "resnet50", 6, img=160, patience=3), "E6 resnet50 lpft"),
    ],
    "lrsearch": [
        (train(f"E7_lr_{tag}", "resnet18", 3, lr=lr, head_lr=lr, img=160, aug="full", ft="full"),
         f"lr {lr}")
        for tag, lr in (("1e-5", 1e-5), ("1e-4", 1e-4), ("3e-4", 3e-4), ("1e-3", 1e-3), ("3e-3", 3e-3))
    ],
    "figure": [
        ([PY, "-m", "ee6483.src.analyze", "--which", "data"], "figs data"),
        ([PY, "-m", "ee6483.src.analyze", "--which", "curves"], "figs curves"),
        ([PY, "-m", "ee6483.src.analyze", "--which", "ablation"], "figs ablation"),
        ([PY, "-m", "ee6483.src.analyze", "--which", "error"], "figs error"),
    ],
}


def already_done(cmd: list[str]) -> bool:
    """Skip a training job whose metrics.json already exists."""
    if "--exp-id" not in cmd:
        return False
    exp = cmd[cmd.index("--exp-id") + 1]
    seed = cmd[cmd.index("--seed") + 1] if "--seed" in cmd else "42"
    return (Path("ee6483/results") / exp / f"s{seed}" / "metrics.json").exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", nargs="+", default=["main", "compare", "ablation", "cifar", "imbalance", "figure"])
    ap.add_argument("--force", action="store_true", help="re-run finished experiments")
    args = ap.parse_args()
    rc_all = 0
    for ph in args.phases:
        print(f"\n########## PHASE {ph} ##########", flush=True)
        for cmd, tag in PHASES[ph]:
            if not args.force and already_done(cmd):
                print(f"skip (done): {tag}", flush=True)
                continue
            rc_all |= job(cmd, f"[{ph}] {tag}")
    print("\nALL DONE", rc_all)


if __name__ == "__main__":
    main()
