"""Shared utilities: seeding, logging, metrics IO."""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------- seeding
def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Fix every RNG we can reach.

    Note: with AMP the CUDA kernels are not bit-exact, so we still report
    mean +- std over multiple seeds instead of claiming bitwise reproducibility.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- logging
class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


class Logger:
    """Prints and appends to a log file at the same time."""

    def __init__(self, path: str | os.PathLike | None = None) -> None:
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                self.path.unlink()

    def __call__(self, msg: str = "") -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}" if msg else ""
        print(line, flush=True)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


# ---------------------------------------------------------------- io
def save_json(obj, path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str | os.PathLike):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: str | os.PathLike) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_args_with_yaml(args, yaml_path: str | None):
    """CLI args override yaml values; yaml fills anything left at default."""
    if not yaml_path:
        return args
    cfg = load_yaml(yaml_path)
    for k, v in cfg.items():
        if hasattr(args, k):
            setattr(args, k, v)
        else:
            setattr(args, k, v)
    return args


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def human_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"
