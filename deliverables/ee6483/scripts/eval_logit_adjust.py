"""Post-hoc logit adjustment evaluated on the CE checkpoint (no retraining).

Logit adjustment (Menon et al., ICLR 2021) changes only the decision rule, so the
"training" half of this method is identical to the CE baseline by construction.
This script loads the CE model, sweeps the adjustment strength tau, and writes a
metrics.json for H_logit_adjust that is directly comparable with the other four
methods (same training, different decision rule).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..src.data.transforms import build_cifar_eval_transform
from ..src.engine import summarise
from ..src.models.backbones import build_cifar_resnet
from ..src.utils import get_device, save_json

CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def get_logits(model, loader, device):
    model.eval()
    logits, tgts = [], []
    with torch.no_grad():
        for imgs, t in loader:
            imgs = imgs.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits.append(model(imgs).float().cpu())
            tgts.append(t)
    return torch.cat(logits), torch.cat(tgts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ce-ckpt", default="ee6483/results/H_ce/s42/best.pt")
    ap.add_argument("--data-root", default="data/cifar")
    ap.add_argument("--out", default="ee6483/results/H_logit_adjust/s42/metrics.json")
    args = ap.parse_args()

    from torch.utils.data import DataLoader

    from ..src.data.cifar_parquet import get_cifar
    from ..src.train_cifar import stratified_split

    device = get_device()
    ck = torch.load(args.ce_ckpt, map_location="cpu", weights_only=False)
    model = build_cifar_resnet("resnet18", 10)
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    # tau is selected on the *validation* split, then applied once to the test set
    from torch.utils.data import Subset

    tr = get_cifar(args.data_root, True, build_cifar_eval_transform())
    _, va_idx = stratified_split(tr.targets, 5000, ck["args"].get("seed", 42))
    val_loader = DataLoader(Subset(tr, va_idx), batch_size=512, shuffle=False, num_workers=4)
    test = get_cifar(args.data_root, False, build_cifar_eval_transform())
    test_loader = DataLoader(test, batch_size=512, shuffle=False, num_workers=4)

    vlog, vtgt = get_logits(model, val_loader, device)
    tlog, ttgt = get_logits(model, test_loader, device)

    ce_metrics = json.loads((Path(args.ce_ckpt).parent / "metrics.json").read_text(encoding="utf-8"))
    counts = ck["args"].get("class_counts") or ce_metrics["class_counts"]
    cnt = torch.tensor([counts[str(c)] for c in range(10)], dtype=torch.float32)
    prior = cnt / cnt.sum()

    print(f"{'tau':>6} {'val acc':>9} {'test acc':>9} {'test mF1':>9}")
    best = None
    for tau in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]:
        vp = F.softmax(vlog - tau * torch.log(prior + 1e-12), 1).numpy()
        v_acc = float((vp.argmax(1) == vtgt.numpy()).mean())
        tp = F.softmax(tlog - tau * torch.log(prior + 1e-12), 1).numpy()
        m = summarise(ttgt.numpy(), tp.argmax(1), tp)
        print(f"{tau:>6} {v_acc*100:>8.2f}% {m['acc']*100:>8.2f}% {m['macro_f1']*100:>8.2f}%")
        if best is None or v_acc > best[0]:
            best = (v_acc, tau, m)

    v_acc, tau, m = best
    out = {
        "exp_id": "H_logit_adjust", "method": "logit_adjust", "mode": "h", "seed": ck["args"].get("seed", 42),
        "best_val_acc": v_acc, "test": m, "class_counts": counts,
        "epochs": ck["args"].get("epochs"), "img_size": 32, "seconds": 0,
        "history": ce_metrics.get("history", []),
        "class_names": CIFAR_CLASSES,
        "note": f"post-hoc logit adjustment on the CE checkpoint; tau={tau} selected on the validation "
                f"split, then applied once to the test set. Training identical to H_ce.",
        "tau": tau,
    }
    save_json(out, args.out)
    print(f"\nselected tau={tau} (val {v_acc*100:.2f}%): test acc {m['acc']*100:.2f}%  "
          f"macroF1 {m['macro_f1']*100:.2f}% -> {args.out}")


if __name__ == "__main__":
    main()
