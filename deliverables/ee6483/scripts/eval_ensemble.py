"""Evaluate the seed-ensemble (+ optional TTA) on the validation set.

Produces ee6483/results/E4_ensemble/metrics.json so the number reported for the
submitted model is measured with exactly the same procedure used to build
submission.csv (average of three seeds, horizontal-flip TTA).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..src.data.dvc_dataset import build_eval_loader
from ..src.engine import summarise
from ..src.predict import load_model
from ..src.utils import get_device, save_json


@torch.no_grad()
def probs_of(model, loader, device, tta=True):
    out = []
    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(imgs)
            if tta:
                logits = (logits + model(torch.flip(imgs, dims=[3]))) / 2
        out.append(F.softmax(logits.float(), 1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", default=[
        "ee6483/results/E4_r18_ft/s42/best.pt",
        "ee6483/results/E4_r18_ft/s43/best.pt",
        "ee6483/results/E4_r18_ft/s44/best.pt",
    ])
    ap.add_argument("--data-root", default="data/cache")
    ap.add_argument("--out", default="ee6483/results/E4_ensemble/metrics.json")
    ap.add_argument("--tta", type=int, default=1)
    args = ap.parse_args()

    device = get_device()
    model0, a0 = load_model(args.ckpts[0], device)
    loader, _ = build_eval_loader(f"{args.data_root}/val", a0["img_size"], 128, 4)

    total, targets = None, None
    for ck in args.ckpts:
        model, _ = load_model(ck, device)
        p = probs_of(model, loader, device, bool(args.tta))
        total = p if total is None else total + p
    probs = total / len(args.ckpts)

    # labels in ImageFolder order (cat, dog)
    targets = np.array(loader.dataset.base.targets)[loader.dataset.indices]
    m = summarise(targets, probs.argmax(1), probs)
    m["n_models"] = len(args.ckpts)
    m["tta"] = bool(args.tta)
    m["n_val"] = int(len(targets))
    m["n_errors"] = int((probs.argmax(1) != targets).sum())
    save_json(m, args.out)
    print(f"ensemble val acc {m['acc']*100:.2f}%  macroF1 {m['macro_f1']*100:.2f}%  "
          f"AUC {m.get('auc')}  errors {m['n_errors']}/{m['n_val']}")
    print("confusion", m["confusion_matrix"])


if __name__ == "__main__":
    main()
