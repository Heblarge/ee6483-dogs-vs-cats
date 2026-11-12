"""测试集预测与 submission.csv 导出。

列名默认为 id,label（cat=0, dog=1）。提交前务必对照课程给出的样例文件核对列名和顺序。
"""
import csv
from pathlib import Path

import torch

from .metrics import predict_logits


def predict(model, loader, device, amp=False, tta_hflip=False):
    logits, ids = predict_logits(model, loader, device, amp, tta_hflip)
    probs = torch.softmax(logits, 1)
    return ids, probs.argmax(1).tolist(), probs[:, 1].tolist()


def write_submission(ids, labels, path, expected_count=500, columns=("id", "label")):
    assert len(ids) == len(set(ids)), "存在重复 id"
    if expected_count is not None:
        assert len(ids) == expected_count, f"预测数 {len(ids)} != {expected_count}"
    rows = sorted(zip(ids, labels), key=lambda r: int(r[0]) if str(r[0]).isdigit() else str(r[0]))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        w.writerows(rows)
    return path
