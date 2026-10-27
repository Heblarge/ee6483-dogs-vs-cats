"""Near-duplicate detection with a *real* Hamming threshold.

`check_data.py` originally compared pHash strings for exact equality, which
under-reports near-duplicates. This script computes 64-bit pHashes for every
train/val image, caches them, and then finds all pairs within a Hamming radius
using blocked numpy popcounts.

    python -m ee6483.scripts.find_duplicates --radius 6
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def hash_one(path_str: str) -> int:
    import imagehash
    from PIL import Image

    with Image.open(path_str) as im:
        return int(str(imagehash.phash(im.convert("L").resize((64, 64)))), 16)


def build_cache(root: Path, cache: Path, workers: int = 8) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Return (hashes, rel_paths, labels) for all train+val images."""
    files, labels = [], []
    for split in ["train", "val"]:
        for ci, cls in enumerate(["cat", "dog"]):
            for p in sorted((root / split / cls).glob("*.jpg")):
                files.append(p)
                labels.append((0 if split == "train" else 1) * 2 + ci)  # split*2 + class
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        if len(d["hashes"]) == len(files):
            print(f"  loaded {len(files)} cached hashes")
            return d["hashes"], [str(x) for x in d["paths"]], d["labels"]
    print(f"  hashing {len(files)} images with {workers} workers ...", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        hashes = np.array(list(ex.map(hash_one, [str(f) for f in files], chunksize=128)), dtype=np.uint64)
    rel = [str(f.relative_to(root)) for f in files]
    labels = np.array(labels, dtype=np.int8)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, hashes=hashes, paths=np.array(rel, dtype=object), labels=labels)
    return hashes, rel, labels


def pairs_within(hashes: np.ndarray, radius: int, block: int = 2048):
    """Yield (i, j) index pairs with popcount(h_i xor h_j) <= radius."""
    n = len(hashes)
    for start in range(0, n, block):
        stop = min(start + block, n)
        hb = hashes[start:stop]
        # compare this block against everything from `start` onwards
        for start2 in range(start, n, block):
            stop2 = min(start2 + block, n)
            hc = hashes[start2:stop2]
            d = np.bitwise_count(np.bitwise_xor(hb[:, None], hc[None, :]))
            ii, jj = np.nonzero(d <= radius)
            for a, b in zip(ii, jj):
                gi, gj = start + int(a), start2 + int(b)
                if gi < gj:
                    yield gi, gj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/datasets")
    ap.add_argument("--cache", default="ee6483/results/phash_cache.npz")
    ap.add_argument("--radius", type=int, default=6)
    ap.add_argument("--out", default="ee6483/results/duplicates.json")
    args = ap.parse_args()

    root = Path(args.root)
    hashes, paths, labels = build_cache(root, Path(args.cache))

    pairs = list(pairs_within(hashes, args.radius))
    print(f"  {len(pairs)} pairs within Hamming <= {args.radius} (of {len(hashes)} images)")

    within_class = cross_class = cross_split = 0
    examples = []
    for i, j in pairs:
        same_split = labels[i] // 2 == labels[j] // 2
        same_class = labels[i] % 2 == labels[j] % 2
        if same_class:
            within_class += 1
        else:
            cross_class += 1
        if not same_split:
            cross_split += 1
        if len(examples) < 20:
            examples.append({"a": paths[i], "b": paths[j],
                             "hamming": int(np.bitwise_count(hashes[i] ^ hashes[j]))})

    # group count = number of connected components over the pair graph
    parent = list(range(len(hashes)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    groups = len({find(i) for i, _ in pairs})

    out = {
        "radius": args.radius,
        "n_images": len(hashes),
        "n_pairs": len(pairs),
        "n_groups": groups,
        "pairs_within_class": within_class,
        "pairs_cross_class": cross_class,
        "pairs_cross_split": cross_split,
        "examples": examples,
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "examples"}, indent=2))


if __name__ == "__main__":
    main()
