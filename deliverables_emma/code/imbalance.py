"""CIFAR-10 long-tailed / class-imbalance experiments (report item h).

We induce imbalance by keeping only ``--minority_size`` samples for a few classes
(default: cat/dog/horse -> 3,5,7). Four training strategies are compared:

    baseline     plain cross-entropy on the imbalanced data
    sampler      WeightedRandomSampler (class-balanced batches)
    weighted_ce  class-weighted cross-entropy
    focal        class-weighted focal loss (gamma=2)

Metrics: overall accuracy, macro-F1, per-class accuracy and minority-class mean
accuracy -- overall accuracy alone hides the damage.

Usage
-----
python imbalance.py --method weighted_ce --out_dir runs/imbalance
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Subset, WeightedRandomSampler, random_split
from torchvision import datasets

from cifar10 import get_transforms
from data import make_loader
from model import Classifier
from trainer import confusion_matrix, evaluate, per_class_accuracy, train_one_epoch


class FocalLoss(nn.Module):
    """Focal loss; ``weight`` (alpha) is optional and applied per class."""

    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer('weight', weight if weight is not None else None)

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction='none')
        pt = torch.exp(-ce)                      # unweighted confidence -> correct focal term
        loss = (1 - pt) ** self.gamma * ce
        if self.weight is not None:
            loss = loss * self.weight[target]
        return loss.mean()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_longtail_indices(targets, minority_classes, minority_size, seed=42):
    targets = np.asarray(targets)
    rng = np.random.RandomState(seed)
    idx = []
    for c in range(10):
        idx_c = np.where(targets == c)[0]
        if c in minority_classes:
            idx_c = rng.choice(idx_c, minority_size, replace=False)
        idx.extend(idx_c.tolist())
    return sorted(idx)


def class_weights(targets, indices, num_classes=10):
    labels = np.asarray(targets)[indices]
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    w = counts.sum() / (num_classes * np.maximum(counts, 1))
    return torch.tensor(w), counts


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default='data/cifar10')
    p.add_argument('--arch', default='resnet18_cifar',
                   choices=['resnet18_cifar', 'resnet18', 'resnet50'])
    p.add_argument('--img_size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--minority_size', type=int, default=500)
    p.add_argument('--minority_classes', default='3,5,7')
    p.add_argument('--method', default='baseline',
                   choices=['baseline', 'sampler', 'weighted_ce', 'focal'])
    p.add_argument('--val_size', type=int, default=5000)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out_dir', default='runs/imbalance')
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    minority = [int(x) for x in args.minority_classes.split(',')]

    train_tf, test_tf = get_transforms(args.img_size)
    train_full = datasets.CIFAR10(args.data_root, train=True, download=False, transform=train_tf)
    n_val = args.val_size
    train_part, val_part = random_split(
        train_full, [len(train_full) - n_val, n_val],
        generator=torch.Generator().manual_seed(args.seed))
    idx_in_train = np.array(train_part.indices if hasattr(train_part, 'indices')
                            else list(range(len(train_full) - n_val)))

    keep = build_longtail_indices(train_full.targets, minority, args.minority_size, args.seed)
    # map global indices -> positions inside train_part
    pos_of = {g: i for i, g in enumerate(idx_in_train)}
    # 关键：把落在 val_part 里的索引剔除，否则 global_labels / sampler 权重
    # 与 Subset 长度不一致 → WeightedRandomSampler 会越界（IndexError）
    keep = [g for g in keep if g in pos_of]
    sub_idx = [pos_of[g] for g in keep]
    train_ds = Subset(train_part, sub_idx)

    val_ds = datasets.CIFAR10(args.data_root, train=True, download=False, transform=test_tf)
    val_ds = Subset(val_ds, val_part.indices)
    test_ds = datasets.CIFAR10(args.data_root, train=False, download=False, transform=test_tf)

    global_labels = np.asarray(train_full.targets)[keep]
    counts = np.bincount(global_labels, minlength=10)
    cw, _ = class_weights(train_full.targets, keep)
    print('Class counts:', counts.tolist())

    if args.method == 'sampler':
        freq = np.maximum(counts, 1)
        sample_w = 1.0 / freq[global_labels]
        sampler = WeightedRandomSampler(torch.DoubleTensor(sample_w),
                                        num_samples=len(sample_w), replacement=True)
        train_ld = make_loader(train_ds, args.batch_size, False, args.num_workers,
                               args.seed, sampler=sampler)
    else:
        train_ld = make_loader(train_ds, args.batch_size, True, args.num_workers, args.seed)
    val_ld = make_loader(val_ds, args.batch_size, False, args.num_workers, args.seed)
    test_ld = make_loader(test_ds, args.batch_size, False, args.num_workers, args.seed)

    pretrained = args.arch in ('resnet18', 'resnet50')
    model = Classifier(args.arch, num_classes=10, pretrained=pretrained, dropout=0.1).to(device)
    if args.method == 'weighted_ce':
        criterion = nn.CrossEntropyLoss(weight=cw.to(device))
    elif args.method == 'focal':
        criterion = FocalLoss(gamma=2.0, weight=cw.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler('cuda', enabled=device == 'cuda')

    best_val, best_state, t0 = 0.0, None, time.time()
    for epoch in range(args.epochs):
        train_one_epoch(model, train_ld, criterion, optimizer, device, scaler=scaler,
                        task='multiclass')
        out = evaluate(model, val_ld, criterion, device, task='multiclass')
        scheduler.step()
        if out['acc'] > best_val:
            best_val = out['acc']
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f'[Epoch {epoch + 1:02d}/{args.epochs}] val_acc={out["acc"]:.4f} '
              f'(best {best_val:.4f})')

    model.load_state_dict(best_state)
    test_out = evaluate(model, test_ld, criterion, device, task='multiclass')
    per_acc, _ = per_class_accuracy(test_out['preds'].numpy(), test_out['targets'].numpy(), 10)
    cm = confusion_matrix(test_out['preds'].numpy(), test_out['targets'].numpy(), 10)
    macro_f1 = f1_score(test_out['targets'].numpy(), test_out['preds'].numpy(),
                        average='macro', zero_division=0)
    np.savetxt(os.path.join(args.out_dir, f'cm_{args.method}.csv'), cm, fmt='%d', delimiter=',')

    result = {'method': args.method, 'best_val_acc': float(best_val),
              'test_acc': float(test_out['acc']), 'macro_f1': float(macro_f1),
              'minority_avg_acc': float(np.mean(per_acc[minority])),
              'per_class_acc': per_acc.tolist(), 'class_counts': counts.tolist(),
              'minority_classes': minority,
              'elapsed_min': round((time.time() - t0) / 60, 1)}
    with open(os.path.join(args.out_dir, f'{args.method}.json'), 'w') as f:
        json.dump(result, f, indent=2)
    torch.save({'model': best_state, 'arch': args.arch, 'img_size': args.img_size},
               os.path.join(args.out_dir, f'{args.method}.pt'))
    print(f'{args.method}: test_acc={test_out["acc"]:.4f} macroF1={macro_f1:.4f} '
          f'minority_avg={result["minority_avg_acc"]:.4f}')


if __name__ == '__main__':
    main()
