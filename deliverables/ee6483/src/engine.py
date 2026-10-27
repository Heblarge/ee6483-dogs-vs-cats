"""Training / evaluation loops shared by every experiment."""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .utils import AverageMeter


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, log_fn=None,
                    grad_clip: float = 1.0, scheduler=None, logit_adjust_counts=None,
                    logit_adjust_tau: float = 1.0):
    model.train()
    loss_m = AverageMeter()
    acc_m = AverageMeter()
    t0 = time.time()

    for step, (imgs, targets) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=scaler is not None):
            logits = model(imgs)
            loss = criterion(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
            optimizer.step()
        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

        bs = targets.size(0)
        loss_m.update(loss.item(), bs)
        acc_m.update((logits.argmax(1) == targets).float().mean().item(), bs)

    if log_fn:
        lr = optimizer.param_groups[0]["lr"]
        log_fn(f"  epoch {epoch:3d} | train loss {loss_m.avg:.4f} acc {acc_m.avg*100:5.2f}% "
               f"| lr {lr:.2e} | {time.time()-t0:.0f}s")
    return {"loss": loss_m.avg, "acc": acc_m.avg}


@torch.no_grad()
def evaluate(model, loader, criterion=None, device=None, return_probs: bool = True,
             tta_hflip: bool = False):
    model.eval()
    device = device or next(model.parameters()).device
    all_logits, all_targets = [], []
    loss_m = AverageMeter()

    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(imgs)
            if tta_hflip:
                logits = (logits + model(torch.flip(imgs, dims=[3]))) / 2
        logits = logits.float()
        if criterion is not None:
            loss_m.update(criterion(logits, targets).item(), targets.size(0))
        all_logits.append(logits.cpu())
        all_targets.append(targets.cpu())

    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets).numpy()
    probs = F.softmax(logits, dim=1).numpy()
    preds = probs.argmax(1)

    metrics = summarise(targets, preds, probs)
    metrics["loss"] = loss_m.avg if criterion is not None else None
    if return_probs:
        metrics["_probs"] = probs
        metrics["_targets"] = targets
        metrics["_logits"] = logits.numpy()
    return metrics


def summarise(targets, preds, probs=None) -> dict:
    n_class = probs.shape[1] if probs is not None else int(max(targets.max(), preds.max())) + 1
    labels = list(range(n_class))
    p, r, f, s = precision_recall_fscore_support(targets, preds, labels=labels, zero_division=0)
    cm = confusion_matrix(targets, preds, labels=labels)
    out = {
        "acc": float(accuracy_score(targets, preds)),
        "balanced_acc": float(balanced_accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro", labels=labels, zero_division=0)),
        "per_class_precision": p.tolist(),
        "per_class_recall": r.tolist(),
        "per_class_f1": f.tolist(),
        "per_class_support": s.tolist(),
        "confusion_matrix": cm.tolist(),
    }
    if probs is not None and n_class == 2:
        try:
            out["auc"] = float(roc_auc_score(targets, probs[:, 1]))
        except ValueError:
            out["auc"] = None
    return out


def strip_private(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if not k.startswith("_")}
