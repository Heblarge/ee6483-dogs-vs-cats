"""Produce review artefacts for the test predictions.

    CUDA_VISIBLE_DEVICES="" python3 review_submission.py --out review

Outputs in --out:
    submission_with_probs.csv   id,label,p_dog
    test_preview.png            contact sheet of the first 24 test images with
                                predicted label + p(dog), for eyeballing
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data import TestDataset, build_eval_transform, make_loader
from model import Classifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default=os.path.expanduser('~/ee6483_data/dataset'))
    ap.add_argument('--ckpt', default='runs/e0_resnet18_160/best.pt')
    ap.add_argument('--out', default='review')
    ap.add_argument('--n', type=int, default=24)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model = Classifier(ckpt['arch'], num_classes=1, pretrained=False, dropout=0.0)
    model.load_state_dict(ckpt['model'])
    model.eval()

    ds = TestDataset(os.path.join(args.data_root, 'test'),
                     build_eval_transform(ckpt['img_size']))
    loader = make_loader(ds, 64, False, 2)

    ids, probs = [], []
    with torch.no_grad():
        for x, img_id in loader:
            logit = model(x)
            logit = (logit + model(torch.flip(x, dims=[3]))) / 2
            probs.extend(torch.sigmoid(logit).squeeze(1).tolist())
            ids.extend(int(i) if i.isdigit() else i for i in img_id)

    df = pd.DataFrame({'id': ids, 'label': [1 if p > 0.5 else 0 for p in probs],
                       'p_dog': np.round(probs, 4)}).sort_values('id')
    df.to_csv(os.path.join(args.out, 'submission_with_probs.csv'), index=False)
    print('rows', len(df), '| dog', int(df.label.sum()), '| uncertain (0.3<p<0.7)',
          int(((df.p_dog > 0.3) & (df.p_dog < 0.7)).sum()))

    # contact sheet: first N images in numeric order
    n = min(args.n, len(df))
    cols = 6
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 3.0 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis('off')
    for ax, (i, row) in zip(axes, df.head(n).iterrows()):
        x, img_id = ds[i]
        img = x.permute(1, 2, 0).numpy() * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(f'id={img_id}  pred={"dog" if row.label else "cat"}  p={row.p_dog:.2f}',
                     fontsize=8)
        ax.axis('on'); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'test_preview.png'), dpi=110, bbox_inches='tight')
    print('saved', args.out)


if __name__ == '__main__':
    main()
