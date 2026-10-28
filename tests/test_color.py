from __future__ import annotations

import numpy as np
import pytest
from src.color import fire_color_mask, smoke_color_mask
from src.preprocessing import Preprocessor


@pytest.fixture
def no_resize_preprocessor() -> Preprocessor:
    return Preprocessor({"resize_width": 0, "blur_ksize": 0})


def test_fire_rgb_rule_selects_orange_pixel(no_resize_preprocessor: Preprocessor):
    frame = np.array([[[0, 140, 255]]], dtype=np.uint8)  # BGR orange
    mask = fire_color_mask(no_resize_preprocessor(frame), {"method": "rgb", "rgb": {"r_threshold": 125, "s_threshold": 60}})
    assert mask.dtype == np.uint8
    assert mask.shape == (1, 1)
    assert mask[0, 0] == 255


def test_fire_combined_rejects_non_fire_colours(no_resize_preprocessor: Preprocessor):
    frame = np.array([[[255, 0, 0], [0, 255, 0], [150, 150, 150]]], dtype=np.uint8)
    mask = fire_color_mask(no_resize_preprocessor(frame), {"method": "combined", "rgb": {"r_threshold": 125, "s_threshold": 60}, "ycbcr": {"tau": 40}})
    assert np.array_equal(mask, np.zeros((1, 3), dtype=np.uint8))


def test_fire_ycbcr_and_combined_select_orange_region(no_resize_preprocessor: Preprocessor):
    # The blue/green background makes frame means meaningful for the YCrCb rule.
    frame = np.array([[[0, 140, 255], [255, 0, 0], [0, 255, 0]]], dtype=np.uint8)
    bundle = no_resize_preprocessor(frame)
    cfg = {"rgb": {"r_threshold": 125, "s_threshold": 60}, "ycbcr": {"tau": 40}}
    ycbcr_mask = fire_color_mask(bundle, {"method": "ycbcr", **cfg})
    combined_mask = fire_color_mask(bundle, {"method": "combined", **cfg})
    assert ycbcr_mask[0, 0] == 255
    assert combined_mask[0, 0] == 255
    assert np.all(combined_mask[0, 1:] == 0)


def test_fire_any_and_and_aliases(no_resize_preprocessor: Preprocessor):
    frame = np.array([[[0, 140, 255], [255, 0, 0], [0, 255, 0]]], dtype=np.uint8)
    bundle = no_resize_preprocessor(frame)
    cfg = {"rgb": {"r_threshold": 125, "s_threshold": 60}, "ycbcr": {"tau": 40}}
    any_mask = fire_color_mask(bundle, {"method": "any", **cfg})
    and_mask = fire_color_mask(bundle, {"method": "and", **cfg})
    assert any_mask.dtype == np.uint8
    assert any_mask.shape == frame.shape[:2]
    assert set(np.unique(any_mask)).issubset({0, 255})
    assert any_mask[0, 0] == 255
    assert and_mask[0, 0] == 255


def test_fire_method_validation(no_resize_preprocessor: Preprocessor):
    bundle = no_resize_preprocessor(np.zeros((2, 2, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="fire_color.method"):
        fire_color_mask(bundle, {"method": "invalid"})


def test_fire_rejects_zero_red_threshold(no_resize_preprocessor: Preprocessor):
    bundle = no_resize_preprocessor(np.zeros((2, 2, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="r_threshold"):
        fire_color_mask(bundle, {"rgb": {"r_threshold": 0}})


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"rgb": {"s_threshold": -1}}, "s_threshold"),
        ({"ycbcr": {"tau": -1}}, "tau"),
    ],
)
def test_fire_rejects_negative_thresholds(no_resize_preprocessor: Preprocessor, config, message):
    bundle = no_resize_preprocessor(np.zeros((2, 2, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match=message):
        fire_color_mask(bundle, config)


def test_smoke_rule_selects_mid_tone_grey(no_resize_preprocessor: Preprocessor):
    frame = np.array([[[150, 148, 152]]], dtype=np.uint8)
    mask = smoke_color_mask(no_resize_preprocessor(frame), {"max_channel_diff": 25, "s_max": 60, "v_min": 80, "v_max": 230})
    assert mask[0, 0] == 255
    assert mask.dtype == np.uint8
    assert mask.shape == frame.shape[:2]
    assert set(np.unique(mask)).issubset({0, 255})


def test_smoke_rule_rejects_saturated_and_out_of_range_pixels(no_resize_preprocessor: Preprocessor):
    frame = np.array([[[0, 140, 255], [40, 40, 40], [245, 245, 245]]], dtype=np.uint8)
    mask = smoke_color_mask(no_resize_preprocessor(frame), {"max_channel_diff": 25, "s_max": 60, "v_min": 80, "v_max": 230})
    assert np.array_equal(mask, np.zeros((1, 3), dtype=np.uint8))


def test_smoke_rejects_invalid_value_range(no_resize_preprocessor: Preprocessor):
    bundle = no_resize_preprocessor(np.zeros((2, 2, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="v_min"):
        smoke_color_mask(bundle, {"v_min": 230, "v_max": 80})
