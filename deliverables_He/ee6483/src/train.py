"""Single-run trainer for the Dogs vs Cats experiments.

Usage example
-------------
python -m ee6483.src.train --model resnet18 --epochs 8 --lr 3e-4 --batch 64 \
    --img-size 224 --aug full --ft-strategy lpft --seed 42 --exp-id E4_r18_ft
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data.dvc_dataset import build_eval_loader, build_train_loader
from .engine import evaluate, strip_private, train_one_epoch
from .models.backbones import (
    build_backbone,
    freeze_backbone,
    head_parameters,
    set_stage_trainable,
    split_backbone_head,
)
from .models.small_cnn import SmallCNN
from .utils import (
    Logger,
    count_parameters,
    get_device,
    human_time,
    merge_args_with_yaml,
    save_json,
    set_seed,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="optional yaml overriding defaults")
    p.add_argument("--data-root", default="data/datasets")
    p.add_argument("--out-root", default="ee6483/results")
    p.add_argument("--exp-id", default="debug")
    p.add_argument("--model", default="resnet18", help="resnet18|resnet34|resnet50|vgg16|smallcnn")
    p.add_argument("--pretrained", type=int, default=1)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4, help="backbone lr")
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--aug", default="full")
    p.add_argument("--ft-strategy", default="lpft",
                   choices=["full", "head", "last_stage", "lpft", "scratch"])
    p.add_argument("--freeze-epochs", type=int, default=2, help="stage-1 length for lpft")
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    p.add_argument("--sched", default="cosine", choices=["cosine", "constant"])
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-per-class", type=int, default=0, help="0 = use all training images")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=4, help="early stop on val acc")
    p.add_argument("--limit-batches", type=int, default=0, help="smoke-test only")
    p.add_argument("--amp", type=int, default=1)
    return p.parse_args(argv)


def build_model(args, num_classes: int = 2):
    pretrained = bool(args.pretrained) and args.ft_strategy != "scratch"
    if args.model == "smallcnn":
        return SmallCNN(num_classes=num_classes), pretrained
    return build_backbone(args.model, num_classes=num_classes, pretrained=pretrained), pretrained


def make_optimizer(args, model):
    backbone, head = split_backbone_head(model)
    groups = []
    if backbone:
        groups.append({"params": backbone, "lr": args.lr, "name": "backbone"})
    groups.append({"params": head, "lr": args.head_lr, "name": "head"})
    if args.optimizer == "adamw":
        return torch.optim.AdamW(groups, weight_decay=args.wd)
    return torch.optim.SGD(groups, momentum=0.9, nesterov=True, weight_decay=args.wd)


def make_scheduler(optimizer, total_steps: int, warmup_steps: int, kind: str = "cosine"):
    def fn(step):
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        if kind == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def main(argv=None):
    args = parse_args(argv)
    args = merge_args_with_yaml(args, args.config)

    set_seed(args.seed)
    device = get_device()
    run_dir = Path(args.out_root) / args.exp_id / f"s{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(run_dir / "train.log")

    log(f"exp={args.exp_id} model={args.model} seed={args.seed} device={device}")
    log(f"args={vars(args)}")

    train_loader, train_base = build_train_loader(
        f"{args.data_root}/train", args.img_size, args.aug, args.batch, args.workers,
        n_per_class=args.n_per_class or None, seed=args.seed,
    )
    val_loader, _ = build_eval_loader(
        f"{args.data_root}/val", args.img_size, batch_size=128, num_workers=args.workers
    )
    log(f"train imgs={len(train_loader.dataset)}  val imgs={len(val_loader.dataset)} "
        f"steps/epoch={len(train_loader)}")

    model, pretrained = build_model(args)
    model.to(device)
    log(f"params={count_parameters(model)/1e6:.2f}M pretrained={pretrained} strategy={args.ft_strategy}")

    # ---- freezing policy -------------------------------------------------
    if args.ft_strategy == "head":
        freeze_backbone(model)
    elif args.ft_strategy == "last_stage":
        set_stage_trainable(model, ("layer4",))
    elif args.ft_strategy == "lpft":
        freeze_backbone(model)  # stage 1; unfrozen after --freeze-epochs

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = make_optimizer(args, model)
    total_epochs = args.epochs
    steps_per_epoch = len(train_loader)
    scheduler = make_scheduler(
        optimizer,
        total_steps=total_epochs * steps_per_epoch,
        warmup_steps=int(args.warmup_epochs * steps_per_epoch),
        kind=args.sched,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp) and device.type == "cuda")

    history = []
    best_acc, best_epoch, bad = -1.0, 0, 0
    t_start = time.time()

    for epoch in range(1, total_epochs + 1):
        if args.ft_strategy == "lpft" and epoch == args.freeze_epochs + 1:
            from .models.backbones import unfreeze_all

            unfreeze_all(model)
            optimizer = make_optimizer(args, model)
            remaining = (total_epochs - args.freeze_epochs) * steps_per_epoch
            scheduler = make_scheduler(
                optimizer, total_steps=remaining,
                warmup_steps=int(0.3 * steps_per_epoch), kind=args.sched,
            )
            log(f"  >> stage 2: backbone unfrozen at epoch {epoch}")

        tr = train_one_epoch(
            model, train_loader, criterion, optimizer,
            scaler if args.amp else None, device,
            epoch, log, args.grad_clip, scheduler,
        )
        va = evaluate(model, val_loader, criterion, device, return_probs=False)
        history.append({"epoch": epoch, "train_loss": tr["loss"], "train_acc": tr["acc"],
                        "val_loss": va["loss"], "val_acc": va["acc"],
                        "val_macro_f1": va["macro_f1"]})
        log(f"          val loss {va['loss']:.4f} acc {va['acc']*100:5.2f}% "
            f"macroF1 {va['macro_f1']*100:5.2f}%")

        if va["acc"] > best_acc:
            best_acc, best_epoch, bad = va["acc"], epoch, 0
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch,
                        "val_acc": va["acc"]}, run_dir / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                log(f"  early stop at epoch {epoch} (best {best_acc*100:.2f}% @ {best_epoch})")
                break

        if args.limit_batches:
            break

    elapsed = time.time() - t_start
    log(f"done in {human_time(elapsed)} | best val acc {best_acc*100:.2f}% @ epoch {best_epoch}")

    with open(run_dir / "curves.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)

    save_json(
        {
            "exp_id": args.exp_id, "seed": args.seed, "model": args.model,
            "img_size": args.img_size, "aug": args.aug, "ft_strategy": args.ft_strategy,
            "epochs_run": len(history), "best_val_acc": best_acc, "best_epoch": best_epoch,
            "params_M": count_parameters(model) / 1e6, "train_seconds": elapsed,
            "n_train": len(train_loader.dataset), "args": vars(args),
        },
        run_dir / "metrics.json",
    )
    return best_acc


if __name__ == "__main__":
    main()
