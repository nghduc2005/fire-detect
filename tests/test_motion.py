from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.detectors.base import Region
from src.motion import MotionDetector
from src.motion.optical_flow import compute_flow, compute_upward_motion
from src.preprocessing import Preprocessor


@pytest.fixture
def preprocessor() -> Preprocessor:
    return Preprocessor({"resize_width": 0, "blur_ksize": 0})


def test_motion_output_shape(preprocessor: Preprocessor):
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    bundle = preprocessor(frame)
    det = MotionDetector({"method": "mog2"})
    mask = det.update(bundle)
    assert mask.shape == (bundle.height, bundle.width)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})


def test_motion_frame_diff_detects_change(preprocessor: Preprocessor):
    det = MotionDetector({"method": "frame_diff", "diff_threshold": 20})
    # Khung f1: tĩnh (toàn 0)
    f1 = np.zeros((60, 80, 3), dtype=np.uint8)
    det.update(preprocessor(f1))
    det.update(preprocessor(f1))

    # Khung f2: xuất hiện vùng sáng (nhấp nháy bật)
    f2 = f1.copy()
    f2[10:30, 10:30] = 255
    det.update(preprocessor(f2))

    # Khung f3: vùng sáng biến mất hoặc di chuyển (nhấp nháy tắt)
    f3 = f1.copy()
    mask = det.update(preprocessor(f3))

    assert mask.dtype == np.uint8
    # Tại vùng (10:30, 10:30), |f2 - f1| = 255 và |f3 - f2| = 255 -> cả 2 đều > 20 -> mask = 255
    assert mask[20, 20] == 255


def test_motion_reset_clears_state(preprocessor: Preprocessor):
    det = MotionDetector({"method": "frame_diff"})
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    det.update(preprocessor(frame))
    det.reset()
    assert det._prev_gray is None
    assert det._prev2_gray is None


def test_motion_all_methods_run(preprocessor: Preprocessor):
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    for method in ("mog2", "knn", "frame_diff", "unknown_fallback"):
        det = MotionDetector({"method": method})
        bundle = preprocessor(frame)
        mask = det.update(bundle)
        assert mask.shape == (bundle.height, bundle.width)
        assert mask.dtype == np.uint8


def test_optical_flow_and_upward_motion():
    # Giả lập 2 frame xám: frame sau dịch chuyển một ô vuông lên trên (y giảm)
    f1 = np.zeros((100, 100), dtype=np.uint8)
    f2 = np.zeros((100, 100), dtype=np.uint8)

    # Ô vuông ở (40..60, 40..60) ở frame 1, chuyển lên (30..50, 40..60) ở frame 2
    f1[40:60, 40:60] = 255
    f2[30:50, 40:60] = 255

    flow = compute_flow(f1, f2)
    assert flow.shape == (100, 100, 2)

    region = Region(bbox=(35, 25, 30, 40), area=400, centroid=(50.0, 45.0), label="smoke")
    upward_ratio = compute_upward_motion(region, flow)
    assert 0.0 <= upward_ratio <= 1.0
    # Chuyển động lên trên rõ ràng nên upward_ratio phải > 0
    assert upward_ratio > 0.0
