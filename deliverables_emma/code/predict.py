"""Generate submission.csv for the 500 test images (report item d).

Usage
-----
python predict.py --ckpt runs/e1_resnet18/best.pt --out submission.csv --tta
"""
import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import TestDataset, build_eval_transform, make_loader
from model import Classifier


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default='dataset')
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out', default='submission.csv')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--tta', action='store_true', help='average original + horizontal flip')
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    arch, img_size = ckpt['arch'], ckpt['img_size']

    model = Classifier(arch, num_classes=1, pretrained=False, dropout=0.0).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    test_ds = TestDataset(os.path.join(args.data_root, 'test'),
                          build_eval_transform(img_size))
    test_ld = make_loader(test_ds, args.batch_size, False, args.num_workers)

    ids, probs = [], []
    for x, img_id in test_ld:
        x = x.to(device, non_blocking=True)
        logit = model(x)
        if args.tta:
            logit = (logit + model(torch.flip(x, dims=[3]))) / 2
        probs.extend(torch.sigmoid(logit).squeeze(1).cpu().tolist())
        ids.extend(int(i) if i.isdigit() else i for i in img_id)

    labels = [1 if p > 0.5 else 0 for p in probs]
    df = pd.DataFrame({'id': ids, 'label': labels}).sort_values('id').reset_index(drop=True)

    # ---- sanity checks required by the brief ----
    assert len(df) == 500, f'expected 500 rows, got {len(df)}'
    assert set(df['label'].unique()) <= {0, 1}
    assert df['id'].min() == 1 and df['id'].max() == 500
    df.to_csv(args.out, index=False)
    print(f'Saved {len(df)} rows -> {args.out} | dog={sum(labels)} cat={500 - sum(labels)} '
          f'| mean prob={sum(probs) / len(probs):.3f}')


if __name__ == '__main__':
    main()
