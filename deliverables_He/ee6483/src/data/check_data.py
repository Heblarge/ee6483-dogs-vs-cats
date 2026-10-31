"""Dataset health check: counts, sizes, corrupt files, id mapping, near-duplicates.

Run before anything else; the report's "Dataset" section quotes these numbers.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def scan(folder: Path, exts=(".jpg", ".jpeg", ".png")):
    files = sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])
    sizes, corrupt = [], []
    for p in files:
        try:
            with Image.open(p) as im:
                sizes.append(im.size)
        except Exception as e:  # noqa: BLE001
            corrupt.append((p.name, str(e)))
    return files, sizes, corrupt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/datasets")
    ap.add_argument("--out", default="ee6483/results/data_report.json")
    ap.add_argument("--hash-dupes", type=int, default=1)
    args = ap.parse_args()

    root = Path(args.root)
    report: dict = {"root": str(root), "splits": {}}

    for split in ["train", "val", "test"]:
        sd = root / split
        entry = {}
        if (sd / "cat").exists():
            for c in ["cat", "dog"]:
                files, sizes, corrupt = scan(sd / c)
                w = np.array([s[0] for s in sizes])
                h = np.array([s[1] for s in sizes])
                entry[c] = {
                    "count": len(files),
                    "corrupt": corrupt,
                    "width": {"min": int(w.min()), "max": int(w.max()), "mean": round(float(w.mean()), 1)},
                    "height": {"min": int(h.min()), "max": int(h.max()), "mean": round(float(h.mean()), 1)},
                    "aspect_ratio_mean": round(float((w / h).mean()), 3),
                    "smallest_side_min": int(np.minimum(w, h).min()),
                }
        else:
            files, sizes, corrupt = scan(sd)
            w = np.array([s[0] for s in sizes])
            h = np.array([s[1] for s in sizes])
            stems = [int(p.stem) for p in files]
            entry = {
                "count": len(files),
                "corrupt": corrupt,
                "width": {"min": int(w.min()), "max": int(w.max()), "mean": round(float(w.mean()), 1)},
                "height": {"min": int(h.min()), "max": int(h.max()), "mean": round(float(h.mean()), 1)},
                "id_min": min(stems),
                "id_max": max(stems),
                "ids_are_contiguous_1_to_n": sorted(stems) == list(range(1, len(stems) + 1)),
                "all_names_numeric": all(p.stem.isdigit() for p in files),
            }
        report["splits"][split] = entry

    # submission cross-check
    sub = Path("sampleSubmission.csv")
    if sub.exists():
        import csv

        with open(sub, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        ids = sorted(int(r["id"]) for r in rows)
        test_ids = sorted(int(p.stem) for p in (root / "test").glob("*.jpg"))
        report["submission"] = {
            "n_rows": len(rows),
            "header": list(rows[0].keys()) if rows else [],
            "ids_match_test_files": ids == test_ids,
            "id_range": [min(ids), max(ids)] if ids else None,
            "label_values_in_sample": sorted(set(r["label"] for r in rows)),
        }

    if args.hash_dupes:
        # Near-duplicate detection lives in scripts/find_duplicates.py, which
        # compares 64-bit pHashes at a Hamming radius (an earlier version of this
        # file only tested for exact hash equality, which under-reports).
        try:
            from ...scripts.find_duplicates import build_cache, pairs_within

            hashes, paths, labels = build_cache(root, Path("ee6483/results/phash_cache.npz"))
            pairs = list(pairs_within(hashes, 6))
            report["near_duplicates"] = {
                "method": "pHash, Hamming <= 6",
                "n_pairs": len(pairs),
                "n_cross_class": sum(1 for i, j in pairs if labels[i] % 2 != labels[j] % 2),
                "n_cross_split": sum(1 for i, j in pairs if labels[i] // 2 != labels[j] // 2),
                "note": "see ee6483/results/duplicates_all.json for the full list",
            }
        except Exception as exc:  # noqa: BLE001
            report["near_duplicates"] = {"error": str(exc)}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    main()
