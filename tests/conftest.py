"""Tiện ích sinh ảnh tổng hợp cho test."""
from __future__ import annotations

import numpy as np
import pytest

from src.config import load_config


@pytest.fixture
def cfg() -> dict:
    return load_config()


@pytest.fixture
def dark_scene() -> np.ndarray:
    """Ảnh nền tối xanh lá 360x640."""
    img = np.zeros((360, 640, 3), np.uint8)
    img[:] = (40, 70, 40)
    return img
