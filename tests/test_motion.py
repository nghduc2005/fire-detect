import cv2
import numpy as np
import pytest

from src.motion import MotionDetector


@pytest.mark.parametrize("method", ["mog2", "knn", "frame_diff"])
def test_moving_square_detected(cfg, dark_scene, method):
    det = MotionDetector({**cfg["motion"], "method": method})
    # Vật có texture: hiệu frame của vật ĐỒNG MÀU bị triệt tiêu ở phần trong (aperture problem)
    patch = np.random.default_rng(0).integers(100, 255, (40, 40, 3), dtype=np.uint8)
    mask = None
    for t in range(40):
        img = dark_scene.copy()
        x = 50 + t * 8
        img[150:190, x:x + 40] = patch
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = det.apply(img, gray)
    x = 50 + 39 * 8
    assert mask[150:190, x:x + 40].mean() > 50
    assert mask[20, 20] == 0


def test_background_available(cfg, dark_scene):
    det = MotionDetector(cfg["motion"])
    gray = cv2.cvtColor(dark_scene, cv2.COLOR_BGR2GRAY)
    det.apply(dark_scene, gray)
    bg = det.background_gray()
    assert bg is not None and bg.shape == gray.shape
    assert np.abs(bg.astype(int) - gray.astype(int)).max() <= 1
