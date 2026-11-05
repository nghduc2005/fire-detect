from __future__ import annotations

import pytest

from src.detectors.base import Region
from src.tracking import RegionTracker, compute_iou


def make_region(x: int, y: int, w: int, h: int, label: str = "fire") -> Region:
    return Region(
        bbox=(x, y, w, h),
        area=w * h,
        centroid=(float(x + w / 2), float(y + h / 2)),
        label=label,
    )


def test_compute_iou():
    box1 = (0, 0, 50, 50)
    box2 = (0, 0, 50, 50)
    assert compute_iou(box1, box2) == 1.0

    box3 = (100, 100, 50, 50)
    assert compute_iou(box1, box3) == 0.0

    # Giao nhau 1 nửa chiều rộng: (0..50) và (25..75) -> giao 25x50 = 1250, hợp = 2500+2500-1250 = 3750 -> 1/3
    box4 = (25, 0, 50, 50)
    assert pytest.approx(compute_iou(box1, box4), rel=1e-3) == 1250.0 / 3750.0


def test_tracker_assigns_unique_ids():
    tracker = RegionTracker()
    regions = [make_region(0, 0, 50, 50), make_region(100, 100, 50, 50)]
    result = tracker.update(regions)
    ids = [r.track_id for r in result]
    assert len(set(ids)) == 2
    assert all(tid > 0 for tid in ids)


def test_tracker_keeps_id_for_overlapping_region():
    tracker = RegionTracker({"iou_threshold": 0.1})
    r1 = [make_region(0, 0, 50, 50)]
    result1 = tracker.update(r1)
    old_id = result1[0].track_id

    # Khung hình sau: vùng di chuyển nhẹ sang (5, 5) -> IoU cao
    r2 = [make_region(5, 5, 50, 50)]
    result2 = tracker.update(r2)
    assert result2[0].track_id == old_id


def test_tracker_history_accumulates():
    tracker = RegionTracker({"buffer_size": 10})
    for i in range(5):
        tracker.update([make_region(i, i, 50, 50)])

    result = tracker.update([make_region(5, 5, 50, 50)])
    assert len(result[0].history) == 6
    assert result[0].history[-1]["bbox"] == (5, 5, 50, 50)


def test_tracker_removes_stale_tracks():
    tracker = RegionTracker({"max_missed": 2})
    tracker.update([make_region(0, 0, 50, 50)])
    assert len(tracker._tracks) == 1

    # Bỏ lỡ 3 khung hình liên tiếp
    tracker.update([])
    tracker.update([])
    tracker.update([])

    # Track quá hạn phải bị xóa khỏi bộ nhớ đệm
    assert len(tracker._tracks) == 0


def test_tracker_reset_clears_state():
    tracker = RegionTracker()
    tracker.update([make_region(0, 0, 50, 50)])
    tracker.reset()
    assert len(tracker._tracks) == 0
    assert len(tracker._history) == 0

