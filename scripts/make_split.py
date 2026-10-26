"""生成固定的子集清单，保证组员使用同一批图像。

python scripts/make_split.py --split train --per-class 1000 --seed 0 --out splits/train_1k.csv
然后在配置中设置 data.train_split: splits/train_1k.csv
"""
import argparse

import _bootstrap  # noqa: F401

from src.data.datasets import list_labeled, save_split, subsample_per_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/datasets")
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--per-class", type=int, default=None, help="不填则导出全量清单")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    samples = list_labeled(args.root, args.split)
    if args.per_class:
        samples = subsample_per_class(samples, args.per_class, args.seed)
    save_split(samples, args.out, args.root)
    print(f"{args.split}: {len(samples)} 张 -> {args.out}")


if __name__ == "__main__":
    main()
