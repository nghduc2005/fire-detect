"""Test cho ``src.config``."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import ConfigDict, DEFAULT_CONFIG_PATH, load_config, load_default_config


def test_configdict_attribute_access():
    cfg = ConfigDict({"a": {"b": {"c": 1}}, "x": [1, 2, 3]})
    assert cfg.a.b.c == 1
    assert cfg["a"]["b"]["c"] == 1
    assert cfg.x == [1, 2, 3]


def test_configdict_get_with_default():
    cfg = ConfigDict({"a": 1})
    assert cfg.get("missing", 42) == 42


def test_configdict_setattr_wraps_dict():
    cfg = ConfigDict()
    cfg.foo = {"bar": 1}
    assert isinstance(cfg.foo, ConfigDict)
    assert cfg.foo.bar == 1


def test_configdict_attribute_error_on_missing():
    cfg = ConfigDict({"a": 1})
    with pytest.raises(AttributeError):
        _ = cfg.nonexistent


def test_load_default_config_file_exists():
    assert DEFAULT_CONFIG_PATH.exists(), (
        f"Thiếu file cấu hình mặc định: {DEFAULT_CONFIG_PATH}"
    )


def test_load_default_config_has_expected_keys():
    cfg = load_default_config()
    # Các khóa gốc được pipeline đọc.
    for key in (
        "pipeline",
        "preprocessing",
        "fire_color",
        "smoke_color",
        "motion",
        "morphology",
        "candidates",
        "tracking",
        "temporal",
        "persistence",
        "scoring",
    ):
        assert key in cfg, f"Thiếu khóa {key!r} trong default.yaml"


def test_load_default_config_values():
    cfg = load_default_config()
    assert cfg.preprocessing.resize_width == 640
    assert cfg.preprocessing.blur_ksize == 5
    assert cfg.fire_color.rgb.r_threshold == 125
    assert cfg.candidates.min_area == 150


def test_load_config_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "no_such.yaml")
