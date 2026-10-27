"""Collect every results/**/metrics.json into one table (Markdown + CSV).

This is the single source of truth for the report's result tables.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def collect(root: Path):
    rows = []
    for mf in sorted(root.rglob("metrics.json")):
        try:
            d = json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "best_val_acc" in d and "test" not in d:  # dogs run
            rows.append({
                "exp": d.get("exp_id"), "seed": d.get("seed"), "model": d.get("model"),
                "img": d.get("img_size"), "aug": d.get("aug"), "ft": d.get("ft_strategy"),
                "n_train": d.get("n_train"), "epochs": d.get("epochs_run"),
                "val_acc": round(d["best_val_acc"] * 100, 2),
                "macro_f1": None, "test_acc": None,
                "sec": round(d.get("train_seconds", 0)),
                "path": str(mf.parent.relative_to(root)),
            })
        elif "test" in d:  # cifar run
            rows.append({
                "exp": d.get("exp_id"), "seed": d.get("seed"), "model": "resnet18-cifar",
                "img": d.get("img_size"), "aug": "-", "ft": d.get("method"),
                "n_train": sum(d.get("class_counts", {}).values()),
                "epochs": d.get("epochs"),
                "val_acc": round(d.get("best_val_acc", 0) * 100, 2),
                "macro_f1": round(d["test"]["macro_f1"] * 100, 2),
                "test_acc": round(d["test"]["acc"] * 100, 2),
                "sec": round(d.get("seconds", 0)),
                "path": str(mf.parent.relative_to(root)),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="ee6483/results")
    ap.add_argument("--out-csv", default="ee6483/results/summary.csv")
    ap.add_argument("--out-md", default="ee6483/results/summary.md")
    args = ap.parse_args()

    rows = collect(Path(args.root))
    if not rows:
        print("no metrics found")
        return
    keys = list(rows[0].keys())
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for r in rows:
        lines.append("| " + " | ".join("" if v is None else str(v) for v in r.values()) + " |")
    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
    print(f"{len(rows)} runs -> {out_csv}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
