from __future__ import annotations

import numpy as np
import pytest

from src.detectors.base import Region
from src.preprocessing import Preprocessor
from src.temporal import FireTemporalAnalyzer
from src.temporal.flicker import (
    compute_fft_flicker,
    compute_intensity_flicker,
    compute_pixel_flicker,
)


@pytest.fixture
def preprocessor() -> Preprocessor:
    return Preprocessor({"resize_width": 0, "blur_ksize": 0})


def test_flicker_functions_empty_history():
    region = Region(bbox=(10, 10, 20, 20), area=400, centroid=(20.0, 20.0), label="fire")
    assert compute_pixel_flicker(region, None) == 0.0
    assert compute_intensity_flicker(region, None) == 0.0
    assert compute_fft_flicker(region) == 0.0


def test_pixel_flicker_oscillating_area():
    region = Region(bbox=(10, 10, 20, 20), area=400, centroid=(20.0, 20.0), label="fire")
    # Giả lập diện tích dao động liên tục: 400 -> 450 -> 390 -> 460 -> 380...
    oscillating_areas = [400, 450, 390, 460, 380, 470, 390, 480, 400]
    region.history = [{"area": a, "bbox": (10, 10, 20, 20)} for a in oscillating_areas]

    score = compute_pixel_flicker(region, None, flicker_window=10)
    assert 0.0 <= score <= 1.0
    assert score > 0.5  # Dao động mạnh liên tục đổi dấu đạo hàm


def test_intensity_flicker_and_fft():
    region = Region(bbox=(10, 10, 20, 20), area=400, centroid=(20.0, 20.0), label="fire")
    # Tạo chuỗi dao động sin tần số 6 Hz (nằm trong dải 3 - 13 Hz của lửa)
    fps = 30.0
    t = np.arange(30) / fps
    sin_wave = 150.0 + 40.0 * np.sin(2 * np.pi * 6.0 * t)

    region.history = [{"intensity_mean": float(val), "area": 400} for val in sin_wave]

    cv_score = compute_intensity_flicker(region, None, min_history=8)
    assert cv_score > 0.0

    fft_score = compute_fft_flicker(region, fps=fps, fire_freq_lo=3.0, fire_freq_hi=13.0, min_history=16)
    # Tần số 6 Hz nằm trọn trong dải [3, 13] Hz nên năng lượng tập trung rất lớn
    assert fft_score > 0.7


def test_fire_temporal_analyzer_update(preprocessor: Preprocessor):
    analyzer = FireTemporalAnalyzer()
    frame = np.full((100, 100, 3), 180, dtype=np.uint8)
    bundle = preprocessor(frame)

    region = Region(bbox=(20, 20, 30, 30), area=900, centroid=(35.0, 35.0), label="fire")
    region.history = [{"bbox": (20, 20, 30, 30), "area": 900}]

    feats = analyzer.update(region, bundle)
    assert "pixel_flicker" in feats
    assert "intensity_flicker" in feats
    assert "fft_flicker" in feats
    assert "intensity_mean" in region.history[-1]
    assert pytest.approx(region.history[-1]["intensity_mean"], rel=1e-2) == 180.0

