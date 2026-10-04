"""Đọc và gộp file cấu hình YAML."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def deep_update(base: dict, override: dict) -> dict:
    """Gộp đệ quy `override` vào bản sao của `base`."""
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict[str, Any]:
    """Đọc config mặc định, sau đó gộp file `path` (nếu có) và `overrides`."""
    with open(DEFAULT_CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG_PATH:
        with open(path, encoding="utf-8") as f:
            cfg = deep_update(cfg, yaml.safe_load(f) or {})
    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg
