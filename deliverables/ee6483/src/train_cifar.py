"""CIFAR-10 experiments: Part G (multi-class transfer) and Part H (class imbalance).

Modes
-----
  g1  ResNet-18 CIFAR-variant trained from scratch (3x3 stem, no maxpool)
  g2  ImageNet-pretrained ResNet-18 on 32->160 upsampled inputs
  h   imbalanced CIFAR-10-LT (rho=100) with one of:
        ce | resample | cb | decouple | logit_adjust

Everything writes metrics.json in the same schema as the Dogs vs Cats runs.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from .data.cifar_parquet import CIFAR_CLASSES, get_cifar

from .data.transforms import (
    build_cifar_eval_transform,
    build_cifar_train_transform,
    build_upsample_transform,
)
from .engine import summarise
from .losses import ClassBalancedLoss, LDAMLoss, logit_adjust
from .models.backbones import build_backbone, build_cifar_resnet, freeze_backbone, unfreeze_all
from .utils import Logger, get_device, human_time, save_json, seed_worker, set_seed


# ------------------------------------------------------------------ data
def stratified_split(targets, val_size: int, seed: int = 42):
    rng = np.random.RandomState(seed)
    tr, va = [], []
    targets = np.asarray(targets)
    per_class = val_size // len(np.unique(targets))
    for c in np.unique(targets):
        idx = np.where(targets == c)[0]
        idx = idx[rng.permutation(len(idx))]
        va.extend(idx[:per_class].tolist())
        tr.extend(idx[per_class:].tolist())
    return sorted(tr), sorted(va)


def long_tailed_indices(targets, rho: float = 100.0, max_count: int = 5000, seed: int = 42):
    """CIFAR-10-LT: n_i = max_count * rho^(-i/(C-1)), classes ordered by index."""
    rng = np.random.RandomState(seed)
    targets = np.asarray(targets)
    classes = np.unique(targets)
    keep = []
    counts = {}
    for i, c in enumerate(classes):
        n = int(round(max_count * rho ** (-i / (len(classes) - 1))))
        idx = np.where(targets == c)[0]
        idx = idx[rng.permutation(len(idx))][:n]
        keep.extend(idx.tolist())
        counts[int(c)] = len(idx)
    return sorted(keep), counts


def loaders(mode: str, root: str, batch: int, workers: int, seed: int,
            imbalanced: bool = False, rho: float = 100.0, img_size: int = 32,
            aug: str = "full", sampler=None):
    if img_size == 32:
        t_tr = build_cifar_train_transform(aug)
        t_ev = build_cifar_eval_transform()
    else:
        t_tr = build_upsample_transform(img_size, train=True, aug=aug)
        t_ev = build_upsample_transform(img_size, train=False)

    base_tr = get_cifar(root, True, t_tr)
    ev_tr = get_cifar(root, True, t_ev)
    test = get_cifar(root, False, t_ev)

    tr_idx, va_idx = stratified_split(base_tr.targets, 5000, seed)
    if imbalanced:
        sub_idx, counts = long_tailed_indices(np.asarray(base_tr.targets)[tr_idx], rho, 5000, seed)
        tr_idx = [tr_idx[i] for i in sub_idx]
        print(f"  long-tailed counts: {counts}  total={len(tr_idx)}")
    else:
        counts = {c: int((np.asarray(base_tr.targets)[tr_idx] == c).sum()) for c in range(10)}

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        Subset(base_tr, tr_idx), batch_size=batch, shuffle=sampler is None, sampler=sampler,
        num_workers=workers, pin_memory=True, drop_last=True,
        worker_init_fn=seed_worker, generator=g, persistent_workers=workers > 0,
    )
    val_loader = DataLoader(Subset(ev_tr, va_idx), batch_size=256, shuffle=False,
                            num_workers=workers, pin_memory=True)
    test_loader = DataLoader(test, batch_size=256, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, val_loader, test_loader, counts


# ------------------------------------------------------------------ train
def evaluate(model, loader, device, adjust_counts=None, tau: float = 1.0):
    model.eval()
    preds, tgts, probs = [], [], []
    with torch.no_grad():
        for imgs, t in loader:
            imgs = imgs.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(imgs).float()
            if adjust_counts is not None:
                logits = logit_adjust(logits, adjust_counts, tau)
            probs.append(F.softmax(logits, 1).cpu().numpy())
            preds.append(logits.argmax(1).cpu().numpy())
            tgts.append(t.numpy())
    return summarise(np.concatenate(tgts), np.concatenate(preds), np.concatenate(probs))


def run(args, method: str):
    set_seed(args.seed)
    device = get_device()
    run_dir = Path(args.out_root) / f"{args.exp_id}" / f"s{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(run_dir / "train.log")

    imbalanced = args.mode == "h"
    sampler = None
    counts = None

    # build model first so we know the input size
    if args.mode == "g2":
        # ImageNet weights, 32 -> 160 upsampled input
        model = build_backbone("resnet18", 10, pretrained=True)
    else:
        # CIFAR variant trained from scratch: g1 (balanced) and h (long-tailed).
        # Pretrained weights would mask the imbalance effect we want to measure.
        model = build_cifar_resnet("resnet18", 10)
    model.to(device)

    train_loader, val_loader, test_loader, counts = loaders(
        args.mode, args.data_root, args.batch, args.workers, args.seed,
        imbalanced=imbalanced, rho=args.rho, img_size=args.img_size, aug=args.aug,
    )
    if method == "resample":
        y = np.array([train_loader.dataset.dataset.targets[i] for i in train_loader.dataset.indices])
        cnt = np.bincount(y, minlength=10).astype(np.float64)
        w = (1.0 / cnt)[y]
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(y), replacement=True,
                                        generator=torch.Generator().manual_seed(args.seed))
        train_loader = DataLoader(train_loader.dataset, batch_size=args.batch, sampler=sampler,
                                  num_workers=args.workers, pin_memory=True, drop_last=True,
                                  worker_init_fn=seed_worker, persistent_workers=args.workers > 0)
        log(f"resample sampler built (class counts {cnt.astype(int).tolist()})")

    log(f"exp={args.exp_id} method={method} mode={args.mode} img={args.img_size} "
        f"train={len(train_loader.dataset)} val={len(val_loader.dataset)}")

    # ---- criterion
    if method == "cb":
        crit = ClassBalancedLoss(list(counts.values()), beta=0.9999, label_smoothing=0.0).to(device)
    elif method == "ldam":
        crit = LDAMLoss(list(counts.values()), max_m=0.5).to(device)
    else:
        crit = nn.CrossEntropyLoss(label_smoothing=0.1 if args.mode != "h" else 0.0)

    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True,
                          weight_decay=args.wd)
    steps = len(train_loader)
    total = args.epochs * steps
    warm = int(0.05 * total)

    def lr_fn(step):
        if step < warm:
            return (step + 1) / warm
        p = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # decoupling: phase-1 boundary
    switch_epoch = int(args.epochs * 0.8) if method == "decouple" else None

    history, best_acc, best_state = [], -1.0, None
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        if switch_epoch and ep == switch_epoch + 1:
            log("  >> decoupling: freeze backbone, re-init head, class-balanced head retrain")
            freeze_backbone(model)
            # re-initialise the head (it has already absorbed the class prior)
            head = model.fc
            linear = head[1] if isinstance(head, nn.Sequential) else head
            nn.init.kaiming_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            # stage 2 sees a class-balanced stream, as in Kang et al. (cRT)
            y2 = np.array([train_loader.dataset.dataset.targets[i] for i in train_loader.dataset.indices])
            cnt2 = np.bincount(y2, minlength=10).astype(np.float64)
            w2 = (1.0 / cnt2)[y2]
            sampler2 = WeightedRandomSampler(
                torch.as_tensor(w2, dtype=torch.double), len(y2), replacement=True,
                generator=torch.Generator().manual_seed(args.seed))
            train_loader = DataLoader(train_loader.dataset, batch_size=args.batch, sampler=sampler2,
                                      num_workers=args.workers, pin_memory=True, drop_last=True,
                                      worker_init_fn=seed_worker, persistent_workers=args.workers > 0)
            steps = len(train_loader)
            opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr * 0.1, momentum=0.9, nesterov=True, weight_decay=args.wd)
            left = (args.epochs - switch_epoch) * steps
            sched = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s / max(1, left), 1.0))))

        model.train()
        tot_loss, n_seen, correct = 0.0, 0, 0
        for imgs, t in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            t = t.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(imgs)
                loss = crit(logits, t)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot_loss += loss.item() * t.size(0)
            n_seen += t.size(0)
            correct += (logits.argmax(1) == t).sum().item()

        va = evaluate(model, val_loader, device)
        history.append({"epoch": ep, "train_loss": tot_loss / n_seen,
                        "train_acc": correct / n_seen, "val_acc": va["acc"],
                        "val_macro_f1": va["macro_f1"]})
        if ep % 5 == 0 or ep == 1 or ep == args.epochs:
            log(f"  ep {ep:3d} | loss {tot_loss/n_seen:.3f} tracc {correct/n_seen*100:5.2f}% "
                f"| val {va['acc']*100:5.2f}% mF1 {va['macro_f1']*100:5.2f}% | {time.time()-t0:.0f}s")
        if va["acc"] > best_acc:
            best_acc = va["acc"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    adj = list(counts.values()) if method == "logit_adjust" else None
    te = evaluate(model, test_loader, device, adjust_counts=adj, tau=1.0)
    log(f"TEST acc {te['acc']*100:.2f}% macroF1 {te['macro_f1']*100:.2f}% "
        f"balanced_acc {te['balanced_acc']*100:.2f}%")

    out = {
        "exp_id": args.exp_id, "method": method, "mode": args.mode, "seed": args.seed,
        "best_val_acc": best_acc, "test": {k: v for k, v in te.items()},
        "class_counts": counts, "epochs": args.epochs, "img_size": args.img_size,
        "seconds": time.time() - t0, "history": history,
        "class_names": CIFAR_CLASSES,
    }
    save_json(out, run_dir / "metrics.json")
    torch.save({"model": model.state_dict(), "args": vars(args), "method": method},
               run_dir / "best.pt")
    log(f"done in {human_time(time.time()-t0)}")
    return te["acc"], te["macro_f1"]


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="g1", choices=["g1", "g2", "h"])
    p.add_argument("--method", default="ce", choices=["ce", "resample", "cb", "ldam", "decouple", "logit_adjust"])
    p.add_argument("--data-root", default="data/cifar")
    p.add_argument("--out-root", default="ee6483/results")
    p.add_argument("--exp-id", default=None)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--img-size", type=int, default=32)
    p.add_argument("--aug", default="full")
    p.add_argument("--rho", type=float, default=100.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args(argv)


if __name__ == "__main__":
    a = parse_args()
    if a.exp_id is None:
        a.exp_id = {"g1": "G1_cifar_scratch", "g2": "G2_cifar_pretrained"}.get(a.mode, f"H_{a.method}")
    if a.mode == "g2":
        a.img_size = 160 if a.img_size == 32 else a.img_size
        a.lr = 0.01 if a.lr == 0.1 else a.lr
        a.epochs = 10 if a.epochs == 80 else a.epochs
    run(a, a.method)
