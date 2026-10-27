"""M0: classical baseline -- HOG + colour histogram -> SVM.

Why this exists: the brief explicitly allows conventional classifiers "as
comparison". It also anchors the bottom of the accuracy ladder, so the gain from
pretrained CNNs is quantified rather than asserted.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def features(path: Path, size: int = 96) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB").resize((size, size), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    g = rgb2gray(arr)
    h = hog(g, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
            block_norm="L2-Hys", feature_vector=True)
    # colour histogram: 16 bins per channel
    hist = np.concatenate([np.histogram(arr[..., c], bins=16, range=(0, 1), density=True)[0]
                           for c in range(3)])
    return np.concatenate([h, hist]).astype(np.float32)


def collect(root: Path, n_per_class: int, seed: int = 42):
    rng = np.random.RandomState(seed)
    X, y = [], []
    for label, cls in enumerate(["cat", "dog"]):
        files = sorted((root / cls).glob("*.jpg"))
        idx = rng.permutation(len(files))[:n_per_class]
        for i in idx:
            X.append(features(files[i]))
            y.append(label)
    return np.stack(X), np.array(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/cache")
    ap.add_argument("--n-train", type=int, default=4000, help="per class")
    ap.add_argument("--n-val", type=int, default=1000, help="per class")
    ap.add_argument("--out", default="ee6483/results/E1_hog_svm/metrics.json")
    args = ap.parse_args()

    t0 = time.time()
    print("extracting train features ...", flush=True)
    Xtr, ytr = collect(Path(args.root) / "train", args.n_train)
    print(f"  {Xtr.shape} in {time.time()-t0:.0f}s", flush=True)
    print("extracting val features ...", flush=True)
    Xva, yva = collect(Path(args.root) / "val", args.n_val)

    results = {}
    for name, clf in {
        "svm_rbf": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", probability=False)),
        "svm_linear": make_pipeline(StandardScaler(), SVC(C=1, kernel="linear")),
    }.items():
        t1 = time.time()
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xva)
        acc = accuracy_score(yva, pred)
        results[name] = {
            "val_acc": float(acc),
            "macro_f1": float(f1_score(yva, pred, average="macro")),
            "confusion_matrix": confusion_matrix(yva, pred).tolist(),
            "fit_seconds": time.time() - t1,
            "feature_dim": int(Xtr.shape[1]),
            "n_train": int(len(ytr)),
            "n_val": int(len(yva)),
        }
        print(f"{name}: val acc {acc*100:.2f}%  ({time.time()-t1:.0f}s)", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
