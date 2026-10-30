"""Offline resize cache: short side -> 256 px, JPEG q=92.

JPEG decode is the dataloader bottleneck on this dataset (images are ~400x400
but we only ever need 224). Caching once cuts epoch time roughly in half and
makes all experiments comparable.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

SPLITS = {"train": ("cat", "dog"), "val": ("cat", "dog")}


def resize_one(job):
    src, dst, short = job
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        if min(w, h) > short:
            scale = short / min(w, h)
            im = im.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.BICUBIC)
        im.save(dst, "JPEG", quality=92)
    return dst.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/datasets")
    ap.add_argument("--dst", default="data/cache")
    ap.add_argument("--short", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    jobs = []
    for split, classes in SPLITS.items():
        for c in classes:
            for p in sorted((Path(args.src) / split / c).glob("*.jpg")):
                jobs.append((p, Path(args.dst) / split / c / p.name, args.short))
    for p in sorted((Path(args.src) / "test").glob("*.jpg")):
        jobs.append((p, Path(args.dst) / "test" / p.name, args.short))

    print(f"caching {len(jobs)} images -> {args.dst} (short side {args.short})")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, _ in enumerate(ex.map(resize_one, jobs, chunksize=64), 1):
            if i % 2000 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
