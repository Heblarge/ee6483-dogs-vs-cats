"""对 test/ 生成 submission.csv，同时保存每张图的 dog 概率便于样例分析。

python scripts/predict.py --ckpt runs/<run_id>/checkpoints/best.pt --out submissions/submission.csv
"""
import argparse
import csv
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from src.data import CLASSES, build_test_loader
from src.evaluation import predict, write_submission
from src.policies import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="submissions/submission.csv")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--expected", type=int, default=500)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    cfg["model"]["pretrained"] = False
    model = build_model(cfg["model"], len(CLASSES))
    model.load_state_dict(ckpt["model"])
    model.to(device)

    ids, labels, p_dog = predict(model, build_test_loader(cfg), device, tta_hflip=args.tta)
    write_submission(ids, labels, args.out, expected_count=args.expected)
    with open(Path(args.out).with_name(Path(args.out).stem + "_probs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "p_dog"])
        w.writerows(zip(ids, labels, [f"{p:.4f}" for p in p_dog]))
    print(f"已写出 {len(ids)} 条预测 -> {args.out}")


if __name__ == "__main__":
    main()
