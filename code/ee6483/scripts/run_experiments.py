"""Sequential experiment orchestrator (one GPU, so no parallelism).

    python -m ee6483.scripts.run_experiments --phases main ablation

Each job is a subprocess so a crash in one run does not kill the queue, and
every job appends its command line to results/run_log.txt for reproducibility.
"""
from __future__ import annotations

import argparse
import json
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
    "augcheck": [
        # Section 7.6: is the 3-epoch augmentation result a budget artefact? 8 epochs, full data and 250/class.
        # Arguments match the saved metrics.json of the original (hand-launched) runs.
        (train("E7b_aug_full_8ep", "resnet18", 8, img=160, aug="full", ft="full", patience=9), "aug full 8ep"),
        (train("E7b_aug_none_8ep", "resnet18", 8, img=160, aug="none", ft="full", patience=9), "aug none 8ep"),
        (train("E7c_aug_full_lowdata", "resnet18", 8, img=160, aug="full", ft="full", n_per_class=250, patience=9),
         "aug full lowdata"),
        (train("E7c_aug_none_lowdata", "resnet18", 8, img=160, aug="none", ft="full", n_per_class=250, patience=9),
         "aug none lowdata"),
    ],
    "cifar": [
        ([PY, "-m", "ee6483.src.train_cifar", "--mode", "g1", "--epochs", "25", "--exp-id", "G1_cifar_scratch"], "G1 cifar scratch"),
        ([PY, "-m", "ee6483.src.train_cifar", "--mode", "g2", "--epochs", "7", "--exp-id", "G2_cifar_pretrained"], "G2 cifar pretrained"),
    ],
    "imbalance": [
        ([PY, "-m", "ee6483.src.train_cifar", "--mode", "h", "--method", m, "--epochs", "18",
          "--exp-id", f"H_{m}"], f"H {m}") for m in ("ce", "resample", "cb", "decouple")
    ] + [
        # longer schedule for the class-balanced loss (the CB row of Section 9.4). The original
        # run was launched by hand and its learning rate was not saved; the default is used here.
        ([PY, "-m", "ee6483.src.train_cifar", "--mode", "h", "--method", "cb", "--epochs", "30",
          "--exp-id", "H_cb_long"], "H cb long"),
        # logit adjustment is post-hoc on the H_ce checkpoint: tau is swept on the validation
        # split and the chosen value is applied once to the test set (no retraining)
        ([PY, "-m", "ee6483.scripts.eval_logit_adjust"], "H logit_adjust (post-hoc on H_ce)"),
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


def _option(cmd: list[str], name: str, default: str) -> str:
    """Return a simple ``--name value`` option from a generated command."""
    return cmd[cmd.index(name) + 1] if name in cmd else default


def expected_artifacts(cmd: list[str]) -> tuple[Path, Path] | None:
    """Return the metrics/checkpoint pair produced by a training command."""
    if "--exp-id" not in cmd:
        return None
    exp = _option(cmd, "--exp-id", "")
    seed = _option(cmd, "--seed", "42")
    root = Path(_option(cmd, "--out-root", "ee6483/results"))
    run_dir = root / exp / f"s{seed}"
    return run_dir / "metrics.json", run_dir / "best.pt"


def already_done(cmd: list[str]) -> bool:
    """Return true only when both usable metrics and a checkpoint exist.

    Metrics are tracked in git while checkpoints are intentionally ignored.  Requiring
    both prevents a fresh clone from skipping training and then failing in evaluation.
    """
    artifacts = expected_artifacts(cmd)
    if artifacts is None:
        return False
    metrics, checkpoint = artifacts
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        return False
    try:
        payload = json.loads(metrics.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and bool(payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", nargs="+", choices=sorted(PHASES),
                    default=["main", "compare", "ablation", "cifar", "imbalance", "figure"])
    ap.add_argument("--force", action="store_true", help="re-run finished experiments")
    args = ap.parse_args()
    rc_all = 0
    for ph in args.phases:
        print(f"\n########## PHASE {ph} ##########", flush=True)
        for cmd, tag in PHASES[ph]:
            if not args.force and already_done(cmd):
                print(f"skip (metrics + checkpoint present): {tag}", flush=True)
                continue
            artifacts = expected_artifacts(cmd)
            if not args.force and artifacts is not None and artifacts[0].exists():
                missing = [str(p) for p in artifacts if not p.exists() or p.stat().st_size == 0]
                if missing:
                    print(f"rerun incomplete job ({', '.join(missing)} missing): {tag}", flush=True)
            rc_all |= job(cmd, f"[{ph}] {tag}")
    print("\nALL DONE", rc_all)
    return rc_all


if __name__ == "__main__":
    raise SystemExit(main())
