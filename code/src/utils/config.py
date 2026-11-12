"""YAML 配置读取，支持命令行 `--set a.b=value` 覆盖。"""
import re
import sys
from pathlib import Path

import yaml

_INT_RE = re.compile(r"[+-]?\d+$")


def _parse_value(raw):
    """解析 --set 的右值。PyYAML 1.1 会把 `3e-4`、`1e-3` 解析成字符串（要求小数点），
    这里对纯数字字符串补一次 int/float 解析，避免覆盖数值字段时得到 str。"""
    v = yaml.safe_load(raw)
    if isinstance(v, str):
        s = v.strip()
        try:
            return int(s) if _INT_RE.match(s) else float(s)
        except ValueError:
            return v
    return v


def _set_by_path(cfg, dotted_key, value):
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
        if not isinstance(node, dict):
            raise TypeError(f"config path '{dotted_key}': '{k}' is not a mapping")
    if keys[-1] not in node:
        # 拼错的键会静默新建、覆盖不生效——至少给出提示
        print(f"WARN: --set '{dotted_key}' not present in config; creating a new key", file=sys.stderr)
    node[keys[-1]] = value


def load_config(path, overrides=None):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for item in overrides or []:
        key, _, raw = item.partition("=")
        _set_by_path(cfg, key.strip(), _parse_value(raw))
    return cfg


def save_config(cfg, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
