"""Ensemble prediction: average sigmoid probabilities of several checkpoints.

Usage:
    python3 predict_ensemble.py --ckpt runs/e0_resnet18_160/best.pt \
                                --ckpt runs/e3_convnext/best.pt \
                                --out submission_ensemble.csv

Each checkpoint carries its own arch / img_size, so the two models are fed with
their own preprocessing — only the probabilities are averaged.
"""
import argparse
import itertools
import os

import pandas as pd
import torch

from data import TestDataset, build_eval_transform, make_loader
from model import Classifier


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default=os.path.expanduser('~/ee6483_data/dataset'))
    p.add_argument('--ckpt', action='append', required=True)
    p.add_argument('--weights', default='', help='comma separated, default equal')
    p.add_argument('--out', default='submission_ensemble.csv')
    p.add_argument('--compare', default='submission.csv')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--tta', action='store_true', default=True)
    return p.parse_args()


@torch.no_grad()
def predict_one(ckpt_path, data_root, batch_size, tta=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = Classifier(ckpt['arch'], num_classes=1, pretrained=False, dropout=0.0).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    ds = TestDataset(os.path.join(data_root, 'test'),
                     build_eval_transform(ckpt['img_size']))
    loader = make_loader(ds, batch_size, False, 4)

    ids, probs = [], []
    for x, img_id in loader:
        x = x.to(device, non_blocking=True)
        logit = model(x)
        if tta:
            logit = (logit + model(torch.flip(x, dims=[3]))) / 2
        probs.extend(torch.sigmoid(logit).squeeze(1).cpu().tolist())
        ids.extend(int(i) if i.isdigit() else i for i in img_id)

    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    return ids, probs, ckpt


def main():
    args = parse_args()
    w = [float(x) for x in args.weights.split(',')] if args.weights else [1.0] * len(args.ckpt)
    assert len(w) == len(args.ckpt)
    w = [x / sum(w) for x in w]

    all_probs, ids0, infos = [], None, []
    for path, weight in zip(args.ckpt, w):
        ids, probs, ckpt = predict_one(path, args.data_root, args.batch_size, args.tta)
        if ids0 is None:
            ids0 = ids
        assert ids == ids0, 'id order mismatch between checkpoints'
        all_probs.append([p * weight for p in probs])
        infos.append(f"{os.path.basename(os.path.dirname(path))}({ckpt['arch']}@{ckpt['img_size']}, w={weight:.2f}, val={ckpt.get('val_acc', 0):.4f})")

    fused = [sum(vals) for vals in zip(*all_probs)]
    labels = [1 if p > 0.5 else 0 for p in fused]
    df = pd.DataFrame({'id': ids0, 'label': labels, 'p_dog': [round(p, 4) for p in fused]}).sort_values('id')
    df[['id', 'label']].to_csv(args.out, index=False)

    print('ensemble of:', ' + '.join(infos))
    print(f'rows={len(df)} dog={int(df.label.sum())} cat={500 - int(df.label.sum())} '
          f'mean p={df.p_dog.mean():.3f}')

    if args.compare and os.path.exists(args.compare):
        old = pd.read_csv(args.compare)
        m = old.merge(df, on='id', suffixes=('_single', '_ens'))
        diff = (m.label_single != m.label_ens).sum()
        print(f'vs {args.compare}: {diff} / {len(m)} labels changed')
        if diff:
            print(m[m.label_single != m.label_ens].head(10).to_string(index=False))
    print('saved', args.out)


if __name__ == '__main__':
    main()
