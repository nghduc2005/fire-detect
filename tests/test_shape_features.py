"""Unit test cho đặc trưng hình dạng lửa (TV4)."""

from __future__ import annotations

import numpy as np

from src.detectors.base import Region
from src.temporal.shape_features import (
    ShapeTemporalAnalyzer,
    compute_area_variation,
    compute_boundary_roughness,
    compute_shape_change,
)


def _region_with_history() -> Region:
    region = Region(bbox=(10, 10, 20, 20), area=400, centroid=(20, 20), label="fire")
    region.mask = np.full((20, 20), 255, dtype=np.uint8)
    for i, (area, dx) in enumerate([(400, 0), (500, 1), (450, 2), (600, 3)]):
        region.history.append(
            {
                "bbox": (10 + dx, 10 + dx, 20, 20),
                "area": area,
                "centroid": (20.0 + dx, 20.0 + dx),
            }
        )
    region.extras["boundary_roughness"] = 1.2
    return region


def test_area_variation_and_shape_change_are_positive() -> None:
    region = _region_with_history()
    assert compute_area_variation(region) > 0.0
    assert compute_shape_change(region) > 0.0


def test_too_short_history_returns_zero() -> None:
    region = Region(bbox=(0, 0, 10, 10), area=100, centroid=(5, 5), label="fire")
    assert compute_area_variation(region) == 0.0
    assert compute_shape_change(region) == 0.0


def test_boundary_roughness_prefers_cached_value() -> None:
    region = _region_with_history()
    assert compute_boundary_roughness(region) == 1.2

    # Không có cache -> tự tính từ mask vuông (perimeter/hull ~ 1.0)
    region.extras.pop("boundary_roughness")
    assert abs(compute_boundary_roughness(region) - 1.0) < 1e-6


def test_shape_analyzer_writes_extras() -> None:
    analyzer = ShapeTemporalAnalyzer({"flicker_window": 10})
    analyzer.reset()
    region = _region_with_history()
    features = analyzer.update(region)
    for key in ("area_variation", "shape_change", "boundary_roughness"):
        assert key in features
        assert key in region.extras
