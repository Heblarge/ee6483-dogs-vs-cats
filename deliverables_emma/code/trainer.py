"""Generic train / evaluate loops (binary + multiclass), with AMP support."""
import contextlib

import numpy as np
import torch
from tqdm import tqdm


def _amp(enabled):
    if enabled:
        return torch.autocast(device_type='cuda', dtype=torch.float16)
    return contextlib.nullcontext()


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None,
                    freeze_backbone=False, task='binary', desc='train'):
    model.train()
    if hasattr(model, 'set_backbone_trainable'):
        model.set_backbone_trainable(not freeze_backbone)

    total_loss, correct, total, skipped = 0.0, 0, 0, 0
    use_amp = scaler is not None
    for x, y in tqdm(loader, desc=desc, leave=False, ncols=80):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _amp(use_amp):
            logits = model(x)
            if task == 'binary':
                loss = criterion(logits, y.float().unsqueeze(1))
            else:
                loss = criterion(logits, y)
        if not torch.isfinite(loss):
            # defensive: a single bad batch must not poison the whole run
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
            continue
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        with torch.no_grad():
            pred = (torch.sigmoid(logits) > 0.5).long().squeeze(1) if task == 'binary' else logits.argmax(1)
            correct += (pred == y).sum().item()
        total += x.size(0)
    if skipped:
        print(f'  [warn] skipped {skipped} non-finite batch(es)')
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, task='binary'):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_preds, all_targets = [], [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        if task == 'binary':
            loss = criterion(logits, y.float().unsqueeze(1))
            prob = torch.sigmoid(logits).squeeze(1)
            pred = (prob > 0.5).long()
            all_probs.append(prob.cpu())
        else:
            loss = criterion(logits, y)
            prob = torch.softmax(logits, dim=1)
            pred = logits.argmax(1)
            all_probs.append(prob.cpu())
        total_loss += loss.item() * x.size(0)
        correct += (pred == y).sum().item()
        total += x.size(0)
        all_preds.append(pred.cpu())
        all_targets.append(y.cpu())
    return {
        'loss': total_loss / max(total, 1),
        'acc': correct / max(total, 1),
        'probs': torch.cat(all_probs),
        'preds': torch.cat(all_preds),
        'targets': torch.cat(all_targets),
    }


def per_class_accuracy(preds, targets, num_classes):
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    correct = np.zeros(num_classes)
    total = np.zeros(num_classes)
    for c in range(num_classes):
        m = targets == c
        total[c] = m.sum()
        correct[c] = (preds[m] == c).sum()
    return correct / np.maximum(total, 1), total


def confusion_matrix(preds, targets, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(np.asarray(targets), np.asarray(preds)):
        cm[t, p] += 1
    return cm
