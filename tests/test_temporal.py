import numpy as np
import pytest

from src.filtering import score_features
from src.filtering.scoring import ramp
from src.structures import Region, bbox_iou
from src.temporal import (PixelFlickerMap, area_variation, temporal_flicker_score,
                          wavelet_flicker_score)
from src.tracking import RegionTracker


def _region(label, x, y, w=40, h=40):
    c = np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]], np.int32)
    return Region(label, (x, y, w, h), float(w * h), float(2 * (w + h)),
                  (x + w / 2, y + h / 2), c)


def test_bbox_iou():
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert bbox_iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0


def test_tracker_keeps_id_and_separates_labels():
    tr = RegionTracker(iou_threshold=0.1, max_missed=2, buffer_size=10)
    tr.update([_region("fire", 10, 10), _region("smoke", 10, 10)])
    tracks = tr.update([_region("fire", 14, 12), _region("smoke", 12, 8)])
    assert len(tracks) == 2
    assert {t.track_id for t in tracks} == {1, 2}
    assert all(len(t.history) == 2 for t in tracks)


def test_tracker_drops_lost_tracks():
    tr = RegionTracker(max_missed=2)
    tr.update([_region("fire", 10, 10)])
    for _ in range(3):
        tracks = tr.update([])
    assert tracks == []


def test_flicker_scores():
    rng = np.random.default_rng(0)
    flicker = rng.normal(100, 20, 32)          # Dao động ngẫu nhiên
    steady = np.linspace(100, 110, 32)          # Thay đổi chậm
    assert temporal_flicker_score(flicker, 25) > temporal_flicker_score(steady, 25)
    assert wavelet_flicker_score(flicker) > wavelet_flicker_score(steady)


def test_pixel_flicker_map_window():
    fm = PixelFlickerMap(window=4)
    on, off = np.full((2, 2), 255, np.uint8), np.zeros((2, 2), np.uint8)
    for i in range(10):
        counts = fm.update(on if i % 2 else off)
    assert counts.max() == 4  # Không vượt quá kích thước cửa sổ


def test_area_variation():
    assert area_variation([100, 100, 100]) == 0.0
    assert area_variation([50, 150, 50, 150]) > 0.4


def test_scoring():
    assert ramp(0.5, 0, 1) == 0.5
    assert ramp(0.8, 1.0, 0.6) == pytest.approx(0.5)  # Đảo chiều
    spec = {"a": [1.0, 0, 1], "b": [1.0, 0, 1]}
    assert score_features({"a": 1.0, "b": None}, spec) == 1.0  # Bỏ qua đặc trưng thiếu
    assert score_features({}, spec) == 0.0
