"""YAML 配置读取，支持命令行 `--set a.b=value` 覆盖。"""
from pathlib import Path

import yaml


def _set_by_path(cfg, dotted_key, value):
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def load_config(path, overrides=None):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for item in overrides or []:
        key, _, raw = item.partition("=")
        _set_by_path(cfg, key.strip(), yaml.safe_load(raw))
    return cfg


def save_config(cfg, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
