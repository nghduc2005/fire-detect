"""Unit test cho SmokeTemporalAnalyzer (đặc trưng khói)."""

from __future__ import annotations

import numpy as np

from src.detectors.base import Region
from src.preprocessing import FrameBundle
from src.temporal.smoke_temporal import SmokeTemporalAnalyzer, lbp_entropy


def _bundle(gray: np.ndarray) -> FrameBundle:
    frame = np.dstack([gray, gray, gray])
    return FrameBundle(
        bgr=frame,
        hsv=np.zeros_like(frame),
        ycrcb=np.zeros_like(frame),
        gray=gray,
    )


def test_lbp_entropy_flat_region_is_zero() -> None:
    flat = np.full((20, 20), 128, dtype=np.uint8)
    assert lbp_entropy(flat) < 0.2


def test_smoke_features_are_written_and_bounded() -> None:
    analyzer = SmokeTemporalAnalyzer(
        {"smoke": {"min_history": 3, "bg_alpha": 0.5, "use_optical_flow": False}}
    )
    region = Region(
        bbox=(10, 10, 40, 40), area=1600, centroid=(30.0, 30.0), label="smoke"
    )
    region.mask = np.full((40, 40), 255, dtype=np.uint8)

    for i in range(3):
        # Nền có vân caro, vùng "khói" là mảng xám phẳng -> ít biên hơn nền
        bg = (np.indices((80, 80)).sum(axis=0) % 2 * 200).astype(np.uint8)
        frame = bg.copy()
        frame[10:50, 10:50] = 130
        region.centroid = (30.0, 30.0 - i)
        region.area = 1600 + 50 * i
        region.history.append(
            {"bbox": region.bbox, "area": region.area, "centroid": region.centroid}
        )
        analyzer.update(region, _bundle(frame))

    for key in ("edge_blur", "texture_smoothness", "upward_motion", "area_growth"):
        assert key in region.extras
    assert 0.0 <= region.extras["edge_blur"] <= 1.5
    assert 0.0 <= region.extras["texture_smoothness"] <= 1.0
    assert region.extras["upward_motion"] > 0
    assert region.extras["area_growth"] > 0


def test_no_history_keeps_motion_features_neutral() -> None:
    analyzer = SmokeTemporalAnalyzer()
    region = Region(bbox=(2, 2, 20, 20), area=400, centroid=(10, 10), label="smoke")
    region.mask = np.full((20, 20), 255, dtype=np.uint8)
    analyzer.update(region, _bundle(np.full((30, 30), 90, dtype=np.uint8)))
    assert "upward_motion" not in region.extras
    assert "area_growth" not in region.extras
    assert "edge_blur" in region.extras


def test_background_is_frozen_inside_region() -> None:
    analyzer = SmokeTemporalAnalyzer({"smoke": {"bg_alpha": 0.5}})
    analyzer.update_background(np.full((20, 20), 100, dtype=np.uint8))

    update_mask = np.full((20, 20), 255, dtype=np.uint8)
    update_mask[5:15, 5:15] = 0
    analyzer.update_background(np.full((20, 20), 200, dtype=np.uint8), update_mask)

    assert analyzer._bg_gray is not None
    assert abs(float(analyzer._bg_gray[5, 5]) - 100.0) < 1e-3   # vùng đóng băng
    assert analyzer._bg_gray[0, 0] > 140.0                      # vùng được cập nhật


def test_smoothing_reduces_frame_to_frame_noise() -> None:
    analyzer = SmokeTemporalAnalyzer({"smoke": {"smooth_window": 3, "use_optical_flow": False}})
    region = Region(bbox=(0, 0, 30, 30), area=900, centroid=(15, 15), label="smoke")
    region.mask = np.full((30, 30), 255, dtype=np.uint8)
    values = []
    for i in range(4):
        gray = np.full((40, 40), 100 + 40 * (i % 2), dtype=np.uint8)
        region.history.append({"bbox": region.bbox, "area": region.area, "centroid": region.centroid})
        analyzer.update(region, _bundle(gray))
        values.append(region.extras["edge_blur"])
    # Làm trơn phải cho giá trị ổn định hơn (không nhảy 0 <-> 1.5)
    assert max(values[1:]) - min(values[1:]) < 0.5
