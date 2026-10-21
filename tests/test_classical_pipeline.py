"""Smoke test tích hợp cho ``ClassicalDetector``.

Mục tiêu: pipeline chạy end-to-end **ngay cả khi** các module của TV2/3/4
mới là stub — không crash, không throw, và các khóa debug mask có mặt.
Khi TV2/3/4 cài đặt thật, test này sẽ tiếp tục pass miễn giữ chữ ký.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import load_default_config
from src.detectors import BaseDetector, ClassicalDetector, Detection, Region


def test_classical_detector_is_a_base_detector():
    assert issubclass(ClassicalDetector, BaseDetector)


def test_pipeline_runs_and_returns_list(random_frame):
    det = ClassicalDetector()
    out = det.process(random_frame)
    assert isinstance(out, list)
    for d in out:
        assert isinstance(d, Detection)


def test_pipeline_debug_masks_populated(random_frame):
    det = ClassicalDetector()
    det.process(random_frame)
    masks = det.debug_masks
    for name in ("fire_color", "smoke_color", "motion", "fire_clean", "smoke_clean"):
        assert name in masks, f"Thiếu mask debug {name!r}"
        assert masks[name].dtype == np.uint8
        assert masks[name].ndim == 2


def test_pipeline_runs_many_frames_without_crash(random_frame):
    """Chạy nhiều frame liên tiếp — kiểm tra trạng thái tracker, motion không crash."""
    det = ClassicalDetector()
    for _ in range(5):
        det.process(random_frame)
    det.reset()
    for _ in range(3):
        det.process(random_frame)


def test_pipeline_respects_ablation_switches(random_frame):
    cfg = load_default_config()
    cfg.pipeline.detect_fire = False
    cfg.pipeline.detect_smoke = False
    det = ClassicalDetector(cfg)
    out = det.process(random_frame)
    assert out == []


def test_detection_dataclass_fields():
    d = Detection(bbox=(1, 2, 3, 4), label="fire", score=0.5)
    assert d.bbox == (1, 2, 3, 4)
    assert d.label == "fire"
    assert d.score == 0.5
    assert d.track_id == -1
    assert d.extras == {}


def test_region_dataclass_fields():
    r = Region(bbox=(0, 0, 10, 10), area=100, centroid=(5.0, 5.0), label="smoke")
    assert r.track_id == -1
    assert r.history == []
    assert r.mask is None
