"""Turn run artefacts into report figures.

    python figures.py --runs runs --out report_figures

Produces:
    curves_<run>.png       train/val loss+acc per epoch
    compare.png            bar chart of best val accuracy across runs
    summary.csv            one row per run (for the report tables)
    cm_<name>.png          confusion-matrix heatmaps found under runs/
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


def plot_curves(history, name, out_dir):
    if not history:
        return
    epochs = [h['epoch'] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(epochs, [h['train_loss'] for h in history], label='train')
    axes[0].plot(epochs, [h['val_loss'] for h in history], label='val')
    axes[0].set_xlabel('epoch'); axes[0].set_ylabel('loss'); axes[0].legend()
    axes[0].set_title(f'{name} — loss')
    axes[1].plot(epochs, [h['train_acc'] for h in history], label='train')
    axes[1].plot(epochs, [h['val_acc'] for h in history], label='val')
    axes[1].set_xlabel('epoch'); axes[1].set_ylabel('accuracy'); axes[1].legend()
    axes[1].set_title(f'{name} — accuracy')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'curves_{name}.png'), dpi=120)
    plt.close(fig)


def plot_cm(path, out_dir, title):
    cm = np.loadtxt(path, delimiter=',', dtype=int)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(cm.shape[1])); ax.set_yticks(range(cm.shape[0]))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax.set_xlabel('predicted'); ax.set_ylabel('true'); ax.set_title(title, fontsize=10)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=7,
                        color='white' if cm[i, j] > cm.max() * 0.5 else 'black')
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'cm_{title.replace(" ", "_")}.png'), dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='runs')
    ap.add_argument('--out', default='report_figures')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for hist_path in sorted(glob.glob(os.path.join(args.runs, '**', 'history.json'),
                                      recursive=True)):
        run_dir = os.path.dirname(hist_path)
        name = os.path.basename(run_dir)
        history = json.load(open(hist_path))
        plot_curves(history, name, args.out)
        res_path = os.path.join(run_dir, 'result.json')
        if os.path.exists(res_path):
            res = json.load(open(res_path))
            rows.append({
                'run': name,
                'arch': res.get('args', {}).get('arch', ''),
                'img_size': res.get('args', {}).get('img_size', ''),
                'lr': res.get('args', {}).get('lr', ''),
                'epochs': res.get('args', {}).get('epochs', ''),
                'strong_aug': res.get('args', {}).get('strong', False),
                'val_acc': res.get('best_val_acc', res.get('test_acc')),
                'test_acc': res.get('test_acc', ''),
                'macro_f1': res.get('macro_f1', ''),
                'minority_avg_acc': res.get('minority_avg_acc', ''),
                'minutes': res.get('elapsed_min', ''),
            })

    if rows:
        df = pd.DataFrame(rows).sort_values('run')
        df.to_csv(os.path.join(args.out, 'summary.csv'), index=False)
        plt.figure(figsize=(max(6, 0.9 * len(df)), 3.8))
        vals = df['val_acc'].astype(float)
        bars = plt.bar(df['run'], vals, color='#4C72B0')
        plt.ylim(max(0, vals.min() - 0.05), 1.0)
        plt.ylabel('best validation accuracy')
        plt.xticks(rotation=30, ha='right')
        for b, v in zip(bars, vals):
            plt.text(b.get_x() + b.get_width() / 2, v + 0.002, f'{v:.4f}',
                     ha='center', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, 'compare.png'), dpi=130)
        plt.close()
        print(df.to_string(index=False))

    for cm_path in sorted(glob.glob(os.path.join(args.runs, '**', '*.csv'), recursive=True)):
        base = os.path.basename(cm_path)
        if base.startswith('cm_') or base == 'confusion_matrix.csv':
            title = os.path.basename(os.path.dirname(cm_path)) + '_' + base.replace('.csv', '')
            plot_cm(cm_path, args.out, title)
    print('Figures written to', args.out)


if __name__ == '__main__':
    main()
