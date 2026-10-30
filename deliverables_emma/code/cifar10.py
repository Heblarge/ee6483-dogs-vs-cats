"""CIFAR-10 classification (report item g).

Key methodological point: the official test set is **never used for model
selection**. We hold out 5,000 images from the 50,000 training images as a
validation split, train on 45,000, select the best epoch on validation, and only
then evaluate once on the 10,000 test images.

Usage
-----
python cifar10.py --arch resnet18 --img_size 32 --epochs 30 --out_dir runs/cifar10
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Subset, random_split
from torchvision import datasets, transforms

from data import CIFAR_MEAN, CIFAR_STD, make_loader
from model import Classifier
from trainer import confusion_matrix, evaluate, per_class_accuracy, train_one_epoch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_transforms(img_size, strong=False):
    # 注意：裁剪尺寸必须跟随 img_size，否则 64px 训练会退化成 32px 裁剪，
    # 与 64px 的验证/测试预处理不一致（会严重掉点）。
    tr = [transforms.RandomCrop(img_size, padding=img_size // 8),
          transforms.RandomHorizontalFlip()]
    if img_size != 32:
        tr = [transforms.Resize(img_size)] + tr
    if strong:
        tr.append(transforms.ColorJitter(0.1, 0.1, 0.1))
    tr += [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]
    te = [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]
    if img_size != 32:
        te = [transforms.Resize(img_size)] + te
    return transforms.Compose(tr), transforms.Compose(te)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default='data/cifar10')
    p.add_argument('--arch', default='resnet18_cifar',
                   choices=['resnet18_cifar', 'resnet18', 'resnet50', 'convnext_tiny'])
    p.add_argument('--img_size', type=int, default=32,
                   help='32 for the CIFAR stem; 64/96 when using an ImageNet backbone')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--backbone_lr_mult', type=float, default=1.0)
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--val_size', type=int, default=5000)
    p.add_argument('--strong', action='store_true')
    p.add_argument('--no_amp', action='store_true')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out_dir', default='runs/cifar10')
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp = (not args.no_amp) and device == 'cuda'

    train_tf, test_tf = get_transforms(args.img_size, args.strong)
    train_full = datasets.CIFAR10(args.data_root, train=True, download=False,
                                  transform=train_tf)
    test_ds = datasets.CIFAR10(args.data_root, train=False, download=False,
                               transform=test_tf)
    n_val = args.val_size
    n_train = len(train_full) - n_val
    train_ds, val_ds = random_split(
        train_full, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed))
    # validation must not be augmented
    val_ds.dataset = datasets.CIFAR10(args.data_root, train=True, download=False,
                                      transform=test_tf)
    print(f'Train {len(train_ds)} | Val {len(val_ds)} | Test {len(test_ds)}')

    train_ld = make_loader(train_ds, args.batch_size, True, args.num_workers, args.seed)
    val_ld = make_loader(val_ds, args.batch_size, False, args.num_workers, args.seed)
    test_ld = make_loader(test_ds, args.batch_size, False, args.num_workers, args.seed)

    pretrained = args.arch in ('resnet18', 'resnet50', 'convnext_tiny')
    model = Classifier(args.arch, num_classes=10, pretrained=pretrained,
                       dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW([
        {'params': model.backbone_parameters(), 'lr': args.lr * args.backbone_lr_mult},
        {'params': model.head_parameters(), 'lr': args.lr},
    ], weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    history, best_acc, best_state, t0 = [], 0.0, None, time.time()
    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_ld, criterion, optimizer, device,
                                          scaler=scaler, task='multiclass')
        out = evaluate(model, val_ld, criterion, device, task='multiclass')
        scheduler.step()
        print(f'[Epoch {epoch + 1:02d}/{args.epochs}] lr={optimizer.param_groups[-1]["lr"]:.2e} '
              f'train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} '
              f'val_loss={out["loss"]:.4f} val_acc={out["acc"]:.4f}')
        history.append({'epoch': epoch + 1, 'train_loss': tr_loss, 'train_acc': tr_acc,
                        'val_loss': out['loss'], 'val_acc': out['acc']})
        if out['acc'] > best_acc:
            best_acc = out['acc']
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            torch.save({'model': best_state, 'arch': args.arch, 'img_size': args.img_size,
                        'val_acc': best_acc}, os.path.join(args.out_dir, 'best.pt'))

    # ---- single evaluation on the untouched test set ----
    model.load_state_dict(best_state)
    test_out = evaluate(model, test_ld, criterion, device, task='multiclass')
    per_acc, counts = per_class_accuracy(test_out['preds'].numpy(),
                                         test_out['targets'].numpy(), 10)
    cm = confusion_matrix(test_out['preds'].numpy(), test_out['targets'].numpy(), 10)
    np.savetxt(os.path.join(args.out_dir, 'confusion_matrix.csv'), cm, fmt='%d', delimiter=',')

    result = {'best_val_acc': float(best_acc), 'test_acc': float(test_out['acc']),
              'per_class_acc': per_acc.tolist(), 'class_counts': counts.tolist(),
              'elapsed_min': round((time.time() - t0) / 60, 1), 'args': vars(args)}
    with open(os.path.join(args.out_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(args.out_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Best val acc {best_acc:.4f} | TEST acc {test_out["acc"]:.4f} '
          f'| {result["elapsed_min"]} min')


if __name__ == '__main__':
    main()
