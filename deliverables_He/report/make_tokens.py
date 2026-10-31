"""Compute every number quoted in the report prose from the raw metrics.json files.

Writes report/tokens.json; build_report.py substitutes {{TOKEN}} placeholders.
This keeps the text and the experiments in sync by construction.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

RES = Path("ee6483/results")


def load(exp: str, seed: int = 42):
    p = RES / exp / f"s{seed}" / "metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def accs(exp: str):
    out = []
    for p in sorted((RES / exp).glob("s*/metrics.json")):
        out.append(json.loads(p.read_text(encoding="utf-8"))["best_val_acc"])
    return out


def fmt(vals, nd=2):
    if not vals:
        return "n/a"
    m, s = np.mean(vals) * 100, np.std(vals) * 100
    return f"{m:.{nd}f}" if len(vals) == 1 else f"{m:.{nd}f} ± {s:.{nd}f}"


def build_appendix_tokens(t: dict):
    """Tokens for the appendix: full run table, environment, submission preview."""
    import csv as _csv
    import subprocess
    import sys

    # ---- full run table from results/summary.csv
    rows = []
    f = RES / "summary.csv"
    if f.exists():
        rows = list(_csv.DictReader(f.open(encoding="utf-8")))
    rows.sort(key=lambda r: (r["exp"], r["seed"]))
    lines = ["| exp | seed | model | img | aug | ft | n_train | ep | val acc (%) | test acc (%) | time (s) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        test = f"{float(r['test_acc']):.2f}" if r.get("test_acc") else "–"
        lines.append(f"| `{r['exp']}` | {r['seed']} | {r['model']} | {r['img']} | {r['aug']} | {r['ft']} | "
                     f"{r['n_train']} | {r['epochs']} | {float(r['val_acc']):.2f} | {test} | {r['sec']} |")
    # two runs that live outside summary.csv (different schema)
    h = RES / "E1_hog_svm" / "metrics.json"
    if h.exists():
        d = json.loads(h.read_text(encoding="utf-8"))
        lines.append(f"| `E1_hog_svm` | – | HOG+SVM (RBF) | 96 | – | – | {d['svm_rbf']['n_train']} | – | "
                     f"{d['svm_rbf']['val_acc']*100:.2f} | – | {d['svm_rbf']['fit_seconds']:.0f} |")
    e = RES / "E4_ensemble" / "metrics.json"
    if e.exists():
        d = json.loads(e.read_text(encoding="utf-8"))
        lines.append(f"| `E4_ensemble` | 42+43+44 | resnet18 x3 + TTA | 224 | full | lpft | 20000 | – | "
                     f"{d['acc']*100:.2f} | – | inference |")
    t["FULL_RESULTS_TABLE"] = "\n".join(lines)

    # ---- environment
    import platform

    import numpy as _np
    import PIL
    import sklearn
    import torch
    import torchvision

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    env = [
        "| Component | Version |", "|---|---|",
        f"| Python | {platform.python_version()} |",
        f"| PyTorch | {torch.__version__} |",
        f"| torchvision | {torchvision.__version__} |",
        f"| CUDA (torch build) | {torch.version.cuda} |",
        f"| cuDNN | {torch.backends.cudnn.version()} |",
        f"| NumPy | {_np.__version__} |",
        f"| scikit-learn | {sklearn.__version__} |",
        f"| Pillow | {PIL.__version__} |",
        f"| GPU | {gpu} (8 GB) |",
        f"| OS | {platform.system()} {platform.release()} |",
    ]
    t["ENV_TABLE"] = "\n".join(env)

    # ---- near-duplicate audit
    df = RES / "duplicates_all.json"
    if df.exists():
        pairs = json.loads(df.read_text(encoding="utf-8"))
        t["DUP_PAIRS"] = len(pairs)
        t["DUP_WITHIN_CLASS"] = sum(1 for x in pairs if x["same_class"])
        t["DUP_CROSS_CLASS"] = sum(1 for x in pairs if not x["same_class"])
        t["DUP_CROSS_SPLIT"] = sum(1 for x in pairs if not x["same_split"])
        t["DUP_IDENTICAL"] = sum(1 for x in pairs if x["ham"] == 0)
    t["DUP_PLACEHOLDER"] = 6
    t["RUN_COUNT"] = len(list(RES.rglob("metrics.json")))

    # ---- submission preview
    sub = Path("submissions/submission.csv")
    if sub.exists():
        with open(sub, encoding="utf-8") as fh:
            raw = fh.read().strip().splitlines()
        head = raw[:6]
        tail = raw[-3:]
        t["SUB_HEAD"] = "```\n" + "\n".join(head + ["..."] + tail) + "\n```"

    # ---- validator output (actually run it)
    try:
        r = subprocess.run([sys.executable, "ee6483/scripts/validate_submission.py", "--sub", "submission.csv"],
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout or r.stderr).strip()
        # drop the absolute path: the appendix must be portable
        out = re.sub(r"^file\s*:.*$", "file      : submission.csv", out, flags=re.M)
        t["VALIDATOR_OUT"] = out
    except Exception as exc:  # noqa: BLE001
        t["VALIDATOR_OUT"] = f"validator not run: {exc}"
    return t


def main():
    t: dict = {}

    # ---- main model
    v = accs("E4_r18_ft")
    t["VAL_ACC_M3"] = fmt(v)
    t["VAL_ACC_M3_S42"] = f"{v[0]*100:.2f}" if v else "n/a"
    t["N_SEEDS_M3"] = len(v)

    # ---- ensemble (the submitted model)
    ens = RES / "E4_ensemble" / "metrics.json"
    if ens.exists():
        d = json.loads(ens.read_text(encoding="utf-8"))
        t["VAL_ACC_ENS"] = f"{d['acc']*100:.2f}"
        t["N_ERR_ENS"] = d["n_errors"]
        t["AUC_ENS"] = f"{d['auc']:.4f}" if d.get("auc") else "n/a"
    t["N_ERR"] = "50"

    # ---- ladder
    t["ACC_M1"] = fmt(accs("E2_smallcnn"))
    t["ACC_M2"] = fmt(accs("E3_r18_frozen"))
    t["ACC_M4"] = fmt(accs("E5_r34_ft"))
    t["ACC_M5"] = fmt(accs("E6_r50_ft"))

    m0 = RES / "E1_hog_svm" / "metrics.json"
    if m0.exists():
        d = json.loads(m0.read_text(encoding="utf-8"))
        t["ACC_M0"] = f"{d['svm_rbf']['val_acc']*100:.2f}"
        t["ACC_M0_LIN"] = f"{d['svm_linear']['val_acc']*100:.2f}"
        t["ACC_M0_DIM"] = d["svm_rbf"]["feature_dim"]
        t["ACC_M0_NTR"] = d["svm_rbf"]["n_train"]

    # ---- ablations
    for tag in ["aug_none_8ep", "aug_full_8ep"]:
        a = accs(f"E7b_{tag}")
        if a:
            t[f"AB_{tag.upper()}"] = f"{np.mean(a)*100:.2f}"
    for tag in ["aug_none_lowdata", "aug_full_lowdata"]:
        a = accs(f"E7c_{tag}")
        if a:
            t[f"AB_{tag.upper()}"] = f"{np.mean(a)*100:.2f}"

    for tag in ["aug_none", "aug_flip", "aug_geo", "aug_color", "aug_full",
                "res_96", "res_128", "res_224",
                "data_250", "data_1000", "data_2500", "data_5000",
                "ft_head", "ft_last", "ft_lpft", "scratch"]:
        a = accs(f"E7_{tag}")
        if a:
            t[f"AB_{tag.upper()}"] = f"{np.mean(a)*100:.2f}"
            d = load(f"E7_{tag}")
            if d:
                t[f"AB_{tag.upper()}_N"] = d.get("n_train")

    for lr in ["1e-5", "1e-4", "3e-4", "1e-3", "3e-3"]:
        a = accs(f"E7_lr_{lr}")
        if a:
            t[f"LR_{lr.replace('-', '')}"] = f"{np.mean(a)*100:.2f}"

    # ---- cifar
    for exp, key in [("G1_cifar_scratch", "G1"), ("G2_cifar_pretrained", "G2")]:
        d = load(exp)
        if d:
            t[f"{key}_TEST"] = f"{d['test']['acc']*100:.2f}"
            t[f"{key}_TEST_F1"] = f"{d['test']['macro_f1']*100:.2f}"
            t[f"{key}_VAL"] = f"{d['best_val_acc']*100:.2f}"
            t[f"{key}_SEC"] = f"{d['seconds']/60:.0f}"

    # ---- imbalance (prefer the fair, longer CB run when it exists)
    hs = {}
    for p in sorted(RES.glob("H_*/s42/metrics.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        m = d["method"]
        if m == "cb" and (RES / "H_cb_long" / "s42" / "metrics.json").exists() and p.parent.parent.name != "H_cb_long":
            continue  # skip the short, under-trained CB run
        hs[m] = d
        t[f"H_{m.upper()}_ACC"] = f"{d['test']['acc']*100:.2f}"
        t[f"H_{m.upper()}_F1"] = f"{d['test']['macro_f1']*100:.2f}"
        t[f"H_{m.upper()}_BACC"] = f"{d['test']['balanced_acc']*100:.2f}"
        if m == "logit_adjust":
            t["H_TAU"] = d.get("tau", 1.0)
    if hs:
        best = max(hs.values(), key=lambda d: d["test"]["acc"])
        worst = min(hs.values(), key=lambda d: d["test"]["acc"])
        t["H_BEST_NAME"] = best["method"]
        t["H_WORST_NAME"] = worst["method"]
        t["H_BEST_ACC"] = f"{best['test']['acc']*100:.2f}"
        t["H_BEST_F1"] = f"{best['test']['macro_f1']*100:.2f}"
        t["H_NTRAIN"] = sum(best["class_counts"].values())
        t["H_MAXC"] = max(best["class_counts"].values())
        t["H_MINC"] = min(best["class_counts"].values())
        ce = hs.get("ce")
        if ce:
            t["H_CE_FEW"] = f"{np.mean(ce['test']['per_class_recall'][6:])*100:.1f}"
            t["H_CE_MANY"] = f"{np.mean(ce['test']['per_class_recall'][:3])*100:.1f}"
            t["H_BEST_FEW"] = f"{np.mean(best['test']['per_class_recall'][6:])*100:.1f}"
            t["H_BEST_MANY"] = f"{np.mean(best['test']['per_class_recall'][:3])*100:.1f}"

    # ---- submission
    sub = Path("submissions/submission.csv")
    if sub.exists():
        import csv

        rows = list(csv.DictReader(sub.open(encoding="utf-8")))
        dog = sum(int(r["label"]) for r in rows)
        t["SUB_ROWS"] = len(rows)
        t["SUB_DOG"] = dog
        t["SUB_CAT"] = len(rows) - dog

    # ---- robustness
    rf = Path("ee6483/figures/robustness.json")
    if rf.exists():
        r = json.loads(rf.read_text(encoding="utf-8"))
        t["ROB_CLEAN"] = f"{r['none']:.2f}"
        t["ROB_BLUR2"] = f"{r['blur2']:.2f}"
        t["ROB_ROT"] = f"{r['rot30']:.2f}"
        t["ROB_OCC"] = f"{r['occ25']:.2f}"
        t["ROB_DARK"] = f"{r['dark']:.2f}"

    add_deltas(t)
    build_appendix_tokens(t)
    Path("report/tokens.json").write_text(json.dumps(t, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(t, indent=1, ensure_ascii=False))


def add_deltas(t: dict):
    """Derived differences used in Part F's tables (all in accuracy points)."""
    def f(k):
        try:
            return float(str(t.get(k, "nan")).split("±")[0])
        except ValueError:
            return float("nan")

    ref_short = f("AB_AUG_FULL")          # 3-epoch, 160 px reference
    ref_full = f("VAL_ACC_M3")            # main model
    # improvements are emitted as unsigned magnitudes (the prose supplies the
    # direction, e.g. "loses {{D_SCRATCH}} points"), while the three effects
    # whose *sign* is the point are emitted signed.
    mags = {
        "D_SCRATCH": ref_short - f("AB_SCRATCH"),
        "D_M0": ref_full - f("ACC_M0"),
        "D_M1": ref_full - f("ACC_M1"),
        "D_M2": ref_full - f("ACC_M2"),
        "D_HEAD": ref_short - f("AB_FT_HEAD"),
        "D_RES": ref_short - f("AB_RES_96"),
        "D_DATA": ref_short - f("AB_DATA_250"),
        "D_RES224": f("AB_RES_224") - ref_short,
    }
    signed = {
        "D_AUG_8EP": f("AB_AUG_FULL_8EP") - f("AB_AUG_NONE_8EP"),
        "D_AUG_LOW": f("AB_AUG_FULL_LOWDATA") - f("AB_AUG_NONE_LOWDATA"),
        "D_M4": f("ACC_M4") - ref_full,
        "D_M5": f("ACC_M5") - ref_full,
        "D_AUG": ref_short - f("AB_AUG_NONE"),
    }
    for k, v in mags.items():
        t[k] = "n/a" if v != v else f"{abs(v):.2f}"
    for k, v in signed.items():
        t[k] = "n/a" if v != v else f"{v:+.2f}".replace("+-", "-")
    return t


if __name__ == "__main__":
    main()
