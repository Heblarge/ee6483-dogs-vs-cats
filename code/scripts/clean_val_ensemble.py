"""近重复敏感性：在去掉与训练集近重复的验证图后，重新计算提交集成的验证准确率。

python scripts/clean_val_ensemble.py \
  --ensemble-json submissions/submission_backbone_ensemble_ensemble.json \
  --duplicates ee6483/results/duplicates_all.json --out results/clean_val.json

只做推理：checkpoint、TTA 设置都取自 ensemble json 的 "chosen" 项，与生成 submission.csv 时完全相同。
被排除的验证图 = duplicates_all.json 中跨 train/val 的近重复对里属于 val 的那一张（去重后计数）。
"""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from src.data import CLASSES
from src.data.datasets import build_val_loader, list_labeled
from src.evaluation.metrics import predict_logits
from src.policies import build_model


def load(path, device, num_workers, data_root):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    cfg["model"]["pretrained"] = False
    cfg["data"]["num_workers"] = num_workers
    if data_root:
        cfg["data"]["root"] = data_root
    model = build_model(cfg["model"], len(CLASSES))
    model.load_state_dict(ckpt["model"])
    return model.to(device), cfg


def cross_split_val_images(dup_path):
    """返回跨 train/val 近重复对中 val 一侧的相对路径集合（形如 val/cat/cat.123.jpg）。"""
    pairs = json.loads(Path(dup_path).read_text(encoding="utf-8"))
    out = set()
    for p in pairs:
        if p["same_split"]:
            continue
        for side in (p["a"], p["b"]):
            rel = side.replace("\\", "/")
            if rel.startswith("val/"):
                out.add(rel)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble-json", default="submissions/submission_backbone_ensemble_ensemble.json")
    ap.add_argument("--duplicates", default="ee6483/results/duplicates_all.json")
    ap.add_argument("--data-root", default="", help="覆盖 checkpoint 配置中的 data.root")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", default="results/clean_val.json")
    args = ap.parse_args()

    chosen = json.loads(Path(args.ensemble_json).read_text(encoding="utf-8"))["chosen"]
    excluded = cross_split_val_images(args.duplicates)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    probs, y, rel = [], None, None
    for ck in chosen["checkpoints"]:
        model, cfg = load(ck, device, args.num_workers, args.data_root)
        logits, yv = predict_logits(model, build_val_loader(cfg), device, amp=True, tta_hflip=chosen["tta"])
        root = Path(cfg["data"]["root"])
        paths = [Path(p).relative_to(root).as_posix() for p, _ in list_labeled(root, "val")]
        assert y is None or (yv == y and paths == rel), "各模型验证集顺序不一致"
        y, rel = yv, paths
        probs.append(logits.softmax(1))
        print(f"{ck}: val acc {100 * (logits.argmax(1) == torch.tensor(yv)).float().mean().item():.2f}%")
        del model
        torch.cuda.empty_cache()

    missing = excluded - set(rel)
    assert not missing, f"近重复列表中的验证图在数据集中找不到: {sorted(missing)}"
    y = torch.tensor(y)
    pred = torch.stack(probs).mean(0).argmax(1)
    keep = torch.tensor([r not in excluded for r in rel])
    wrong = pred != y
    report = {
        "checkpoints": chosen["checkpoints"], "tta": chosen["tta"],
        "n_val": len(y), "val_acc": round(100 * (~wrong).float().mean().item(), 2),
        "val_errors": int(wrong.sum()),
        "n_excluded": int((~keep).sum()), "excluded": sorted(excluded),
        "excluded_errors": [r for r, k, w in zip(rel, keep.tolist(), wrong.tolist()) if not k and w],
        "n_clean": int(keep.sum()),
        "clean_val_acc": round(100 * (~wrong[keep]).float().mean().item(), 2),
        "clean_val_errors": int(wrong[keep].sum()),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("checkpoints", "excluded")}, indent=2))


if __name__ == "__main__":
    main()
