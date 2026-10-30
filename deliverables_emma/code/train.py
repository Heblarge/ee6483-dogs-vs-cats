"""Dogs vs Cats training (report items b, c, d).

Example
-------
python train.py --arch resnet18 --img_size 160 --epochs 15 --batch_size 32 \
                --out_dir runs/e1_resnet18
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets

from data import (build_eval_transform, get_dataloaders, make_loader)
from model import Classifier
from trainer import evaluate, train_one_epoch


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default='dataset')
    p.add_argument('--arch', default='resnet18',
                   choices=['resnet18', 'resnet50', 'convnext_tiny'])
    p.add_argument('--img_size', type=int, default=160)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--backbone_lr_mult', type=float, default=0.1)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--freeze_epochs', type=int, default=1)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--strong', action='store_true')
    p.add_argument('--no_pretrained', action='store_true')
    p.add_argument('--no_amp', action='store_true')
    p.add_argument('--max_train_per_class', type=int, default=None)
    p.add_argument('--max_val_per_class', type=int, default=None)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--deterministic', action='store_true')
    p.add_argument('--out_dir', default='runs/baseline')
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed, args.deterministic)
    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device, '| AMP:', not args.no_amp and device == 'cuda')

    train_ld, val_ld, _, train_ds, val_ds, class_names = get_dataloaders(
        args.data_root, args.img_size, args.batch_size, args.num_workers,
        args.strong, args.max_train_per_class, args.max_val_per_class, args.seed)
    print(f'Train: {len(train_ds)} | Val: {len(val_ds)} | classes: {class_names}')

    model = Classifier(args.arch, num_classes=1, pretrained=not args.no_pretrained,
                       dropout=args.dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW([
        {'params': model.backbone_parameters(), 'lr': args.lr * args.backbone_lr_mult},
        {'params': model.head_parameters(), 'lr': args.lr},
    ], weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler('cuda', enabled=(not args.no_amp and device == 'cuda'))

    history, best_acc, t0 = [], 0.0, time.time()
    for epoch in range(args.epochs):
        freeze = epoch < args.freeze_epochs
        tr_loss, tr_acc = train_one_epoch(model, train_ld, criterion, optimizer, device,
                                          scaler=scaler, freeze_backbone=freeze)
        out = evaluate(model, val_ld, criterion, device)
        scheduler.step()
        lr_now = optimizer.param_groups[-1]['lr']
        print(f'[Epoch {epoch + 1:02d}/{args.epochs}] lr={lr_now:.2e} '
              f'train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} '
              f'val_loss={out["loss"]:.4f} val_acc={out["acc"]:.4f}')
        history.append({'epoch': epoch + 1, 'train_loss': tr_loss, 'train_acc': tr_acc,
                        'val_loss': out['loss'], 'val_acc': out['acc'], 'lr': lr_now})
        if out['acc'] > best_acc:
            best_acc = out['acc']
            torch.save({'model': model.state_dict(), 'arch': args.arch,
                        'img_size': args.img_size, 'val_acc': out['acc'],
                        'class_names': class_names}, os.path.join(args.out_dir, 'best.pt'))

    # ---- final metrics on the best checkpoint (val set) ----
    ckpt = torch.load(os.path.join(args.out_dir, 'best.pt'), map_location=device,
                      weights_only=False)
    model.load_state_dict(ckpt['model'])
    out = evaluate(model, val_ld, criterion, device)
    precision, recall, f1, _ = precision_recall_fscore_support(
        out['targets'].numpy(), out['preds'].numpy(), average='binary', zero_division=0)

    result = {
        'best_val_acc': float(best_acc),
        'final_val_acc': float(out['acc']),
        'precision': float(precision), 'recall': float(recall), 'f1': float(f1),
        'train_size': len(train_ds), 'val_size': len(val_ds),
        'elapsed_min': round((time.time() - t0) / 60, 1),
        'args': vars(args),
    }
    with open(os.path.join(args.out_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(args.out_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Best val acc: {best_acc:.4f} | P/R/F1: {precision:.4f}/{recall:.4f}/{f1:.4f} '
          f'| {result["elapsed_min"]} min')


if __name__ == '__main__':
    main()
