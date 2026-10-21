"""
Tiền xử lý frame.

Pipeline tiền xử lý được chuẩn hóa cho mọi detector:

1. **Resize giữ tỉ lệ** về một chiều rộng chuẩn (vd. 640px) để
   các ngưỡng diện tích, kích thước đều ổn định giữa các video.
2. **Gaussian blur** 5×5 để giảm nhiễu cảm biến trước khi phân tích màu
   / chuyển động.
3. **Chuyển không gian màu** sang HSV và YCbCr vì:
   - Luật màu lửa dùng cả RGB (Chen 2004) và YCbCr (Çelik 2009).
   - Luật màu khói cần HSV (bão hòa, độ sáng).

Kết quả gom vào :class:`FrameBundle` — các tầng sau không cần tự chuyển lại.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import cv2
import numpy as np

__all__ = [
    "FrameBundle",
    "Preprocessor",
    "resize_keep_aspect",
    "gaussian_blur",
    "to_hsv",
    "to_ycrcb",
    "to_gray",
]


# ---------------------------------------------------------------------- #
# Hàm tiện ích (dạng stateless, dễ unit test)
# ---------------------------------------------------------------------- #
def resize_keep_aspect(frame: np.ndarray, target_width: Optional[int]) -> np.ndarray:
    """
    Resize giữ tỉ lệ về chiều rộng chuẩn.

    Nếu ``target_width`` là ``None`` hoặc ``<= 0`` thì giữ nguyên frame.
    """
    if target_width is None or target_width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w == target_width:
        return frame
    scale = float(target_width) / float(w)
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame, (int(target_width), new_h), interpolation=cv2.INTER_AREA)


def gaussian_blur(frame: np.ndarray, ksize: Optional[int]) -> np.ndarray:
    """
    Gaussian blur với kernel vuông ``(k, k)``.

    - ``ksize <= 0`` hoặc ``None`` : tắt (trả về nguyên frame).
    - ``ksize`` chẵn               : tự động cộng thêm 1 cho hợp lệ.
    """
    if ksize is None or ksize <= 0:
        return frame
    k = int(ksize)
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(frame, (k, k), 0)


def to_hsv(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)


def to_ycrcb(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)


def to_gray(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------- #
# Gói các biểu diễn của một frame
# ---------------------------------------------------------------------- #
@dataclass
class FrameBundle:
    """Các biểu diễn của một frame sau tiền xử lý.

    Attributes
    ----------
    bgr : np.ndarray
        Frame đã resize + blur, dạng BGR uint8 (H, W, 3).
    hsv : np.ndarray
        Frame ``bgr`` chuyển sang HSV.
    ycrcb : np.ndarray
        Frame ``bgr`` chuyển sang YCbCr (OpenCV: thứ tự kênh là Y, Cr, Cb).
    gray : np.ndarray
        Frame ``bgr`` chuyển sang grayscale.
    """

    bgr: np.ndarray
    hsv: np.ndarray
    ycrcb: np.ndarray
    gray: np.ndarray

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.bgr.shape  # type: ignore[return-value]

    @property
    def height(self) -> int:
        return int(self.bgr.shape[0])

    @property
    def width(self) -> int:
        return int(self.bgr.shape[1])

    @property
    def area(self) -> int:
        return self.height * self.width


# ---------------------------------------------------------------------- #
# Lớp Preprocessor — áp dụng pipeline theo cấu hình
# ---------------------------------------------------------------------- #
class Preprocessor:
    """Preprocessor dùng chung cho mọi detector.

    Parameters
    ----------
    config : Mapping[str, Any] | None
        Toàn bộ ConfigDict của dự án hoặc chỉ phần ``preprocessing``.
        Các khóa đọc được:

        - ``preprocessing.resize_width`` (int) — mặc định 640.
        - ``preprocessing.blur_ksize``   (int) — mặc định 5, số lẻ.

        Nếu ``config=None`` thì dùng giá trị mặc định.
    """

    DEFAULT_RESIZE_WIDTH: int = 640
    DEFAULT_BLUR_KSIZE: int = 5

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        # Hỗ trợ cả truyền toàn bộ config lẫn chỉ phần "preprocessing".
        pre: Mapping[str, Any] = {}
        if config is not None:
            if "preprocessing" in config and isinstance(config["preprocessing"], Mapping):
                pre = config["preprocessing"]
            else:
                pre = config

        self.resize_width: int = int(pre.get("resize_width", self.DEFAULT_RESIZE_WIDTH))
        self.blur_ksize: int = int(pre.get("blur_ksize", self.DEFAULT_BLUR_KSIZE))

    # ------------------------------------------------------------------ #
    def __call__(self, frame_bgr: np.ndarray) -> FrameBundle:
        """Áp dụng pipeline tiền xử lý lên 1 frame và trả về :class:`FrameBundle`."""
        if frame_bgr is None:
            raise ValueError("Preprocessor nhận frame=None.")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(
                f"Preprocessor cần frame BGR (H, W, 3), nhận được shape={frame_bgr.shape}"
            )

        img = resize_keep_aspect(frame_bgr, self.resize_width)
        img = gaussian_blur(img, self.blur_ksize)

        return FrameBundle(
            bgr=img,
            hsv=to_hsv(img),
            ycrcb=to_ycrcb(img),
            gray=to_gray(img),
        )

    # Tiện lợi: cho phép dùng ``preprocessor.apply(frame)`` tường minh.
    def apply(self, frame_bgr: np.ndarray) -> FrameBundle:
        return self(frame_bgr)
