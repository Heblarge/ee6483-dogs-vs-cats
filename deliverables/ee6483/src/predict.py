"""Generate submission.csv from one or more checkpoints (ensemble + hflip TTA).

Example
-------
python -m ee6483.src.predict \
    --ckpts ee6483/results/E4_r18_ft/s42/best.pt ee6483/results/E4_r18_ft/s43/best.pt \
            ee6483/results/E4_r18_ft/s44/best.pt \
    --out submission.csv --tta 1
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data.dvc_dataset import build_test_loader
from .models.backbones import build_backbone
from .models.small_cnn import SmallCNN
from .utils import get_device


def load_model(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    if a["model"] == "smallcnn":
        model = SmallCNN(num_classes=2)
    else:
        model = build_backbone(a["model"], num_classes=2, pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, a


@torch.no_grad()
def predict_probs(model, loader, device, tta: bool = True):
    probs, ids = [], []
    for imgs, batch_ids in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(imgs)
            if tta:
                logits = logits + model(torch.flip(imgs, dims=[3]))
        if tta:
            logits = logits / 2
        probs.append(F.softmax(logits.float(), dim=1).cpu().numpy())
        ids.extend(batch_ids.tolist())
    return np.concatenate(probs), np.array(ids)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--data-root", default="data/datasets")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--tta", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)

    device = get_device()
    model0, a0 = load_model(args.ckpts[0], device)
    loader = build_test_loader(f"{args.data_root}/test", a0["img_size"], 128, args.workers)

    total = None
    ids = None
    for i, ck in enumerate(args.ckpts):
        model, a = load_model(ck, device) if i else (model0, a0)
        p, ids = predict_probs(model, loader, device, bool(args.tta))
        print(f"  {Path(ck).parent}  mean p(dog)={p[:,1].mean():.3f}")
        total = p if total is None else total + p
    probs = total / len(args.ckpts)
    preds = probs.argmax(1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for i, p in zip(ids, preds):
            w.writerow([int(i), int(p)])

    n_dog = int(preds.sum())
    print(f"wrote {out}  rows={len(preds)}  dog={n_dog}  cat={len(preds)-n_dog}  "
          f"mean confidence={probs.max(1).mean():.4f}")
    np.save(out.with_suffix(".probs.npy"), probs)


if __name__ == "__main__":
    main()
