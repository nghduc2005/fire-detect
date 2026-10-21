"""Test cho ``src.preprocessing``."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import load_default_config
from src.preprocessing import (
    FrameBundle,
    Preprocessor,
    gaussian_blur,
    resize_keep_aspect,
    to_gray,
    to_hsv,
    to_ycrcb,
)


def test_resize_keep_aspect(random_frame):
    out = resize_keep_aspect(random_frame, 320)
    assert out.shape[1] == 320
    # Tỉ lệ phải giữ nguyên (sai số ±1 pixel chấp nhận do làm tròn).
    expected_h = round(random_frame.shape[0] * 320 / random_frame.shape[1])
    assert abs(out.shape[0] - expected_h) <= 1


def test_resize_noop_when_target_none(random_frame):
    assert resize_keep_aspect(random_frame, None) is random_frame
    assert resize_keep_aspect(random_frame, 0) is random_frame
    assert resize_keep_aspect(random_frame, -1) is random_frame


def test_resize_noop_when_same_width(random_frame):
    out = resize_keep_aspect(random_frame, random_frame.shape[1])
    assert out is random_frame


def test_gaussian_blur_odd_kernel(random_frame):
    out = gaussian_blur(random_frame, 5)
    assert out.shape == random_frame.shape


def test_gaussian_blur_even_kernel_handled(random_frame):
    # ksize=4 phải tự tăng lên 5 chứ không crash.
    out = gaussian_blur(random_frame, 4)
    assert out.shape == random_frame.shape


def test_gaussian_blur_disabled(random_frame):
    assert gaussian_blur(random_frame, 0) is random_frame
    assert gaussian_blur(random_frame, None) is random_frame


def test_color_space_conversions(random_frame):
    hsv = to_hsv(random_frame)
    ycrcb = to_ycrcb(random_frame)
    gray = to_gray(random_frame)
    assert hsv.shape == random_frame.shape
    assert ycrcb.shape == random_frame.shape
    assert gray.ndim == 2 and gray.shape == random_frame.shape[:2]


def test_preprocessor_from_config(random_frame):
    cfg = load_default_config()
    pre = Preprocessor(cfg)
    bundle = pre(random_frame)

    assert isinstance(bundle, FrameBundle)
    # Phải resize đúng chiều rộng config.
    assert bundle.width == cfg.preprocessing.resize_width
    # Tỉ lệ aspect giữ nguyên (±1 pixel).
    expected_h = round(random_frame.shape[0] * cfg.preprocessing.resize_width / random_frame.shape[1])
    assert abs(bundle.height - expected_h) <= 1
    # Các biểu diễn phải đồng kích thước.
    assert bundle.hsv.shape == bundle.bgr.shape
    assert bundle.ycrcb.shape == bundle.bgr.shape
    assert bundle.gray.shape == bundle.bgr.shape[:2]


def test_preprocessor_defaults(random_frame):
    pre = Preprocessor()
    bundle = pre(random_frame)
    assert bundle.width == Preprocessor.DEFAULT_RESIZE_WIDTH


def test_preprocessor_rejects_bad_input():
    pre = Preprocessor()
    with pytest.raises(ValueError):
        pre(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        pre(np.zeros((10, 10), dtype=np.uint8))  # 2D không hợp lệ
