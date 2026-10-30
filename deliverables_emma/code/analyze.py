"""Error analysis on the validation set (report item e).

The upload test set has no labels, so all quantitative analysis is done on the
validation split; this is stated explicitly in the report.

Outputs (in --out_dir):
    wrong_samples.csv    path, true label, predicted label, p(dog)
    wrong_grid.png       grid of the most confident mistakes
    right_grid.png       grid of correctly classified samples
    confidence_hist.png  p(dog) distribution for the two classes
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision import datasets

from data import build_eval_transform, make_loader
from model import Classifier

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def denorm(t):
    img = t.permute(1, 2, 0).numpy() * STD + MEAN
    return np.clip(img, 0, 1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default='dataset')
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out_dir', default='analysis')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--max_wrong', type=int, default=12)
    p.add_argument('--max_right', type=int, default=12)
    return p.parse_args()


def grid(records, ds, title, path, cols=6):
    n = len(records)
    if n == 0:
        return
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.6 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis('off')
    for ax, (idx, prob, true, pred) in zip(axes, records):
        x, _ = ds[idx]
        ax.imshow(denorm(x))
        ax.set_title(f'true={ds.classes[true]} pred={ds.classes[pred]}\np={prob:.3f}', fontsize=8)
        ax.axis('on')
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = Classifier(ckpt['arch'], num_classes=1, pretrained=False, dropout=0.0).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    ds = datasets.ImageFolder(os.path.join(args.data_root, 'val'),
                             transform=build_eval_transform(ckpt['img_size']))
    loader = make_loader(ds, args.batch_size, False, 4)

    probs = []
    for x, _ in loader:
        probs.extend(torch.sigmoid(model(x.to(device))).squeeze(1).cpu().tolist())
    probs = np.array(probs)
    targets = np.array(ds.targets)
    preds = (probs > 0.5).astype(int)

    rows = []
    for i, (path, _) in enumerate(ds.samples):
        rows.append({'file': os.path.basename(path), 'true': int(targets[i]),
                     'pred': int(preds[i]), 'p_dog': float(probs[i]),
                     'correct': bool(preds[i] == targets[i])})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, 'wrong_samples.csv'), index=False)

    wrong = df[~df['correct']].copy()
    # most confident mistakes first
    wrong['conf'] = (wrong['p_dog'] - 0.5).abs()
    wrong = wrong.sort_values('conf', ascending=False)
    print(f'Val acc={df["correct"].mean():.4f} | wrong={len(wrong)}/{len(df)}')

    # filename -> position in the dataset (needed to re-read the image)
    pos = {os.path.basename(p): i for i, (p, _) in enumerate(ds.samples)}
    wrong_recs = [(pos[r.file], r.p_dog, r.true, r.pred)
                  for r in wrong.head(args.max_wrong).itertuples()]
    right = df[df['correct']].sort_values('p_dog', ascending=False)
    right_recs = [(pos[r.file], r.p_dog, r.true, r.pred)
                  for r in right.head(args.max_right).itertuples()]

    grid(wrong_recs, ds, 'Misclassified validation samples', 
         os.path.join(args.out_dir, 'wrong_grid.png'))
    grid(right_recs, ds, 'Correctly classified validation samples (high confidence)',
         os.path.join(args.out_dir, 'right_grid.png'))

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(probs[targets == 0], bins=30, alpha=0.6, label='cat (0)')
    ax.hist(probs[targets == 1], bins=30, alpha=0.6, label='dog (1)')
    ax.axvline(0.5, color='k', ls='--', lw=1)
    ax.set_xlabel('p(dog)'); ax.set_ylabel('count'); ax.legend()
    ax.set_title('Validation confidence distribution')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'confidence_hist.png'), dpi=110)
    plt.close(fig)
    print('Saved:', os.listdir(args.out_dir))


if __name__ == '__main__':
    main()
