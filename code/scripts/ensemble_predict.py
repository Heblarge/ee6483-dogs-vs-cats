"""多个 checkpoint 集成：先在验证集上比较各候选方案，再用选定方案生成 submission.csv。

python scripts/ensemble_predict.py --out submissions/submission.csv \
  --group convnext=runs/backbones/b_convnext_tiny_s0/checkpoints/best.pt,runs/backbones/b_convnext_tiny_s1/checkpoints/best.pt \
  --group top3=<6 个 checkpoint，逗号分隔> --choose top3

集成方式：对各模型的 softmax 概率取平均。每个方案都分别报告不带 / 带水平翻转 TTA 的验证准确率。
候选方案要在看结果之前定好，--choose 指定最终采用哪一个，避免在验证集上反复挑选。
"""
import argparse
import csv
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from src.data import CLASSES, build_test_loader
from src.data.datasets import build_val_loader
from src.evaluation import write_submission
from src.evaluation.metrics import predict_logits
from src.policies import build_model


def load(path, device, num_workers):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    cfg["model"]["pretrained"] = False
    cfg["data"]["num_workers"] = num_workers
    model = build_model(cfg["model"], len(CLASSES))
    model.load_state_dict(ckpt["model"])
    return model.to(device), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append", required=True, help="名字=ckpt1,ckpt2,...")
    ap.add_argument("--choose", required=True, help="最终采用的方案名")
    ap.add_argument("--tta", action="store_true", help="最终方案使用水平翻转 TTA")
    ap.add_argument("--out", default="submissions/submission.csv")
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    groups = dict(g.split("=", 1) for g in args.group)
    groups = {k: v.split(",") for k, v in groups.items()}
    assert args.choose in groups, f"--choose {args.choose} 不在 {list(groups)}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 每个 checkpoint 只跑一次：验证集（不带/带 TTA）和测试集（不带/带 TTA）的概率
    probs, y_val, test_ids = {}, None, None
    for ck in sorted({c for cs in groups.values() for c in cs}):
        model, cfg = load(ck, device, args.num_workers)
        out = {}
        for tta in (False, True):
            lv, yv = predict_logits(model, build_val_loader(cfg), device, amp=True, tta_hflip=tta)
            lt, ids = predict_logits(model, build_test_loader(cfg), device, amp=True, tta_hflip=tta)
            out[("val", tta)], out[("test", tta)] = lv.softmax(1), lt.softmax(1)
            assert y_val is None or yv == y_val, "各模型验证集顺序不一致"
            assert test_ids is None or ids == test_ids, "各模型测试集顺序不一致"
            y_val, test_ids = yv, ids
        probs[ck] = out
        acc = (out[("val", False)].argmax(1) == torch.tensor(y_val)).float().mean().item()
        print(f"{ck}: val acc {100 * acc:.2f}%")
        del model
        torch.cuda.empty_cache()

    y = torch.tensor(y_val)
    report = {}
    for name, cks in groups.items():
        for tta in (False, True):
            p = torch.stack([probs[c][("val", tta)] for c in cks]).mean(0)
            pred = p.argmax(1)
            report[f"{name}{'+tta' if tta else ''}"] = {
                "val_acc": round(100 * (pred == y).float().mean().item(), 2),
                "val_errors": int((pred != y).sum()),
                "n_models": len(cks),
            }
    for k, v in report.items():
        print(f"{k:20s} val acc {v['val_acc']:.2f}%  errors {v['val_errors']}  ({v['n_models']} models)")

    cks = groups[args.choose]
    p = torch.stack([probs[c][("test", args.tta)] for c in cks]).mean(0)
    labels = p.argmax(1).tolist()
    write_submission(test_ids, labels, args.out)
    out = Path(args.out)
    with open(out.with_name(out.stem + "_probs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "p_dog"])
        w.writerows(zip(test_ids, labels, [f"{x:.4f}" for x in p[:, 1].tolist()]))
    report["chosen"] = {"group": args.choose, "tta": args.tta, "checkpoints": cks}
    out.with_name(out.stem + "_ensemble.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"采用 {args.choose}{'+tta' if args.tta else ''}：已写出 {len(test_ids)} 条预测 -> {out}"
          f"（dog {sum(labels)} / cat {len(labels) - sum(labels)}）")


if __name__ == "__main__":
    main()
