"""在验证集上评估 checkpoint，输出指标 JSON 与误分类清单（用于样例分析）。

python scripts/evaluate.py --ckpt runs/<run_id>/checkpoints/best.pt [--tta]
"""
import argparse
import csv
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from src.data import CLASSES
from src.data.datasets import build_val_loader
from src.evaluation import evaluate_model
from src.policies import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tta", action="store_true", help="水平翻转测试时增强")
    ap.add_argument("--out", default=None, help="默认写到 checkpoint 所在 run 目录")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    cfg["model"]["pretrained"] = False  # 权重来自 checkpoint，无需再下载
    model = build_model(cfg["model"], len(CLASSES))
    model.load_state_dict(ckpt["model"])
    model.to(device)

    loader = build_val_loader(cfg)
    metrics = evaluate_model(model, loader, device, tta_hflip=args.tta)
    probs = metrics.pop("probs")

    out = Path(args.out or Path(args.ckpt).parents[1] / "eval")
    out.mkdir(parents=True, exist_ok=True)
    (out / "val_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    with open(out / "val_misclassified.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "true", "pred", "p_dog"])
        for (path, y), p in zip(loader.dataset.samples, probs):
            pred = int(p.argmax())
            if pred != y:
                w.writerow([path, CLASSES[y], CLASSES[pred], f"{p[1]:.4f}"])
    print(f"val_acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} -> {out}")


if __name__ == "__main__":
    main()
