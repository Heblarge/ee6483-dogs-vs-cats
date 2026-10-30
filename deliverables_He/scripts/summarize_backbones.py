"""汇总 runs/backbones/ 下的骨干对比结果，输出到 results/backbones.{csv,md}。

python scripts/summarize_backbones.py [--runs runs/backbones]

同时报告 best（按验证集挑 epoch）和 last（最后一个 epoch）准确率：两者差距大说明结果依赖挑选 epoch。
"""
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml

from src.data import CLASSES
from src.policies import build_model


def read_run(run_dir):
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    with open(run_dir / "history.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    acc = [float(r["val_accuracy"]) for r in rows]
    return {
        "model": cfg["model"]["name"],
        "seed": cfg.get("seed", 0),
        "epochs_done": len(rows),
        "epochs_cfg": cfg["train"]["epochs"],
        "best": max(acc),
        "best_epoch": acc.index(max(acc)) + 1,
        "last": acc[-1],
        "sec_per_epoch": statistics.mean(float(r["seconds"]) for r in rows),
    }


def n_params(name):
    model = build_model({"name": name, "pretrained": False}, len(CLASSES))
    return sum(p.numel() for p in model.parameters()) / 1e6


def mean_std(xs):
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/backbones")
    ap.add_argument("--out", default="results/backbones")
    args = ap.parse_args()

    by_model = defaultdict(list)
    for d in sorted(Path(args.runs).iterdir()):
        if d.is_dir() and (d / "history.csv").exists():
            r = read_run(d)
            if r:
                by_model[r["model"]].append(r)

    table = []
    for name, runs in by_model.items():
        complete = [r for r in runs if r["epochs_done"] >= r["epochs_cfg"]]
        if len(complete) < len(runs):
            print(f"注意：{name} 有 {len(runs) - len(complete)} 个未跑完的 run，已忽略")
        if not complete:
            continue
        b_m, b_s = mean_std([r["best"] * 100 for r in complete])
        l_m, l_s = mean_std([r["last"] * 100 for r in complete])
        table.append({
            "model": name, "params_M": round(n_params(name), 2), "seeds": len(complete),
            "best_acc": round(b_m, 2), "best_std": round(b_s, 2),
            "last_acc": round(l_m, 2), "last_std": round(l_s, 2),
            "best_epochs": "/".join(str(r["best_epoch"]) for r in complete),
            "sec_per_epoch": round(statistics.mean(r["sec_per_epoch"] for r in complete)),
        })
    table.sort(key=lambda r: r["params_M"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]) if table else ["model"])
        w.writeheader()
        w.writerows(table)
    lines = ["| 模型 | 参数量 (M) | seeds | best val acc (%) | last val acc (%) | best epoch | s/epoch |",
             "|---|---:|---:|---:|---:|---|---:|"]
    for r in table:
        lines.append(f"| {r['model']} | {r['params_M']} | {r['seeds']} | {r['best_acc']} ± {r['best_std']} | "
                     f"{r['last_acc']} ± {r['last_std']} | {r['best_epochs']} | {r['sec_per_epoch']} |")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
