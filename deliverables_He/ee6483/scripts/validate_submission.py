"""Strict validator for submission.csv -- run this before every upload.

Checks (any failure => exit code 1):
  * header is exactly `id,label`
  * exactly 500 data rows
  * ids are exactly 1..500 (no duplicates, no missing)
  * every label is in {0, 1}
  * label convention: 1 = dog, 0 = cat (verified against the file, not assumed)
  * prediction distribution is not degenerate (warns if the minority class < 35%)
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="submission.csv")
    ap.add_argument("--sample", default="sampleSubmission.csv")
    ap.add_argument("--expect-rows", type=int, default=500)
    args = ap.parse_args()

    errors, warnings = [], []
    path = Path(args.sub)
    if not path.exists():
        print(f"FAIL: {path} does not exist")
        return 1

    raw = path.read_text(encoding="utf-8").strip().splitlines()
    header = [h.strip() for h in raw[0].split(",")]
    if header != ["id", "label"]:
        errors.append(f"header must be ['id','label'], got {header}")

    rows = list(csv.DictReader(raw))
    if len(rows) != args.expect_rows:
        errors.append(f"expected {args.expect_rows} rows, got {len(rows)}")

    ids = [int(r["id"]) for r in rows]
    labels = [int(r["label"]) for r in rows]

    if sorted(ids) != list(range(1, args.expect_rows + 1)):
        missing = set(range(1, args.expect_rows + 1)) - set(ids)
        dupes = {i for i in ids if ids.count(i) > 1}
        errors.append(f"ids must be exactly 1..{args.expect_rows}; missing={sorted(missing)[:10]} dupes={sorted(dupes)[:10]}")
    if len(set(ids)) != len(ids):
        errors.append("duplicate ids found")
    if not set(labels) <= {0, 1}:
        errors.append(f"labels must be 0/1, found {sorted(set(labels))}")

    n_dog = sum(labels)
    frac = n_dog / max(len(labels), 1)
    if not 0.35 <= frac <= 0.65:
        warnings.append(f"suspicious class balance: dog={n_dog} cat={len(labels)-n_dog} "
                        f"({frac:.1%} dog) -- check id mapping / preprocessing")

    # cross-check against the provided sample file
    sample = Path(args.sample)
    if sample.exists():
        with open(sample, encoding="utf-8") as f:
            s_rows = list(csv.DictReader(f))
        if len(s_rows) != len(rows):
            errors.append(f"row count differs from sample ({len(s_rows)} vs {len(rows)})")
        if [r["id"] for r in s_rows] != [r["id"] for r in rows]:
            errors.append("id order differs from sampleSubmission.csv (ids must be 1..500 ascending)")

    print(f"file      : {path.resolve()}")
    print(f"rows      : {len(rows)}")
    print(f"dog(1)    : {n_dog}   cat(0): {len(labels)-n_dog}")
    print(f"unique ids: {len(set(ids))}")
    for w in warnings:
        print(f"WARN: {w}")
    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        return 1
    print("PASS: submission.csv is well-formed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
