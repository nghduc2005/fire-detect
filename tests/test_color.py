import numpy as np

from src.color import fire_color_mask, smoke_color_mask
from src.preprocessing import preprocess


def test_fire_patch_detected(cfg, dark_scene):
    img = dark_scene.copy()
    img[100:200, 100:200] = (20, 140, 250)  # Cam (BGR)
    pf = preprocess(img, {"resize_width": 640, "blur_ksize": 0})
    mask = fire_color_mask(pf, cfg["fire_color"])
    assert mask[150, 150] == 255
    assert mask[10, 10] == 0


def test_blue_not_fire(cfg, dark_scene):
    img = dark_scene.copy()
    img[100:200, 100:200] = (250, 120, 20)  # Xanh dương
    pf = preprocess(img, {"resize_width": 640, "blur_ksize": 0})
    assert fire_color_mask(pf, cfg["fire_color"]).max() == 0


def test_gray_is_smoke_color(cfg, dark_scene):
    img = dark_scene.copy()
    img[50:150, 300:400] = (170, 172, 168)
    pf = preprocess(img, {"resize_width": 640, "blur_ksize": 0})
    mask = smoke_color_mask(pf, cfg["smoke_color"])
    assert mask[100, 350] == 255
    assert mask[300, 50] == 0  # Nền xanh lá


def test_mask_shape_and_dtype(cfg, dark_scene):
    pf = preprocess(dark_scene, cfg["preprocessing"])
    for m in (fire_color_mask(pf, cfg["fire_color"]), smoke_color_mask(pf, cfg["smoke_color"])):
        assert m.dtype == np.uint8
        assert m.shape == pf.shape
