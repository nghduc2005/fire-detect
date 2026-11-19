"""Đặc trưng khói theo không gian & thời gian — Thành viên 4.

Bốn đặc trưng khớp với ``configs/default.yaml`` → ``scoring.smoke``:

- ``edge_blur``          : tỷ lệ năng lượng biên Sobel so với ảnh nền xám
  (nhỏ hơn 1 nghĩa là vùng mờ hơn nền — đặc trưng của khói).
- ``texture_smoothness`` : độ mịn kết cấu từ entropy mã LBP 8 láng giềng.
- ``upward_motion``      : tỷ lệ pixel chuyển động hướng lên (Optical Flow
  Farnebäck của TV3); nếu không có flow thì fallback theo dịch chuyển trọng tâm.
- ``area_growth``        : tốc độ nở diện tích vùng khói theo thời gian.

Các đặc trưng được làm trơn trượt để giảm nhiễu từng frame. Nền xám được cập
nhật trượt nhưng **đóng băng bên trong vùng khói** để không hấp thụ chính đối
tượng cần đo.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any, Deque, Dict, Iterable, Mapping, Optional

import cv2
import numpy as np

from ..detectors.base import Region

if TYPE_CHECKING:  # tránh vòng import khi nạp package
    from ..motion.optical_flow import compute_flow, compute_upward_motion

__all__ = ["SmokeTemporalAnalyzer", "lbp_entropy"]


def _fast_uniform_lbp_entropy(
    gray_roi: np.ndarray, mask_roi: Optional[np.ndarray] = None
) -> float:
    """Entropy của mã LBP 8 láng giềng (thuần NumPy/OpenCV, không cần scikit-image)."""
    if gray_roi.shape[0] < 3 or gray_roi.shape[1] < 3:
        return 2.5
    center = gray_roi[1:-1, 1:-1]
    neighbors = [
        gray_roi[0:-2, 0:-2],
        gray_roi[0:-2, 1:-1],
        gray_roi[0:-2, 2:],
        gray_roi[1:-1, 2:],
        gray_roi[2:, 2:],
        gray_roi[2:, 1:-1],
        gray_roi[2:, 0:-2],
        gray_roi[1:-1, 0:-2],
    ]
    lbp = np.zeros_like(center, dtype=np.uint8)
    for idx, nb in enumerate(neighbors):
        lbp |= ((nb >= center).astype(np.uint8) << idx)
    if mask_roi is not None:
        valid_mask = mask_roi[1:-1, 1:-1] > 0
        vals = lbp[valid_mask]
    else:
        vals = lbp.ravel()
    if vals.size == 0:
        return 2.5
    hist = np.bincount(vals, minlength=256).astype(np.float64)
    hist /= (hist.sum() + 1e-9)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


# Alias công khai để tái sử dụng / unit test
lbp_entropy = _fast_uniform_lbp_entropy


class SmokeTemporalAnalyzer:
    """Trích xuất đặc trưng khói cho :class:`~src.filtering.FeatureScorer`."""

    def __init__(self, cfg: Optional[Mapping[str, Any]] = None) -> None:
        cfg = cfg or {}
        smoke_cfg = cfg.get("smoke", {})
        if not isinstance(smoke_cfg, Mapping):
            smoke_cfg = {}

        def pick(key: str, default: Any) -> Any:
            return smoke_cfg.get(key, cfg.get(key, default))

        self.min_history = int(pick("min_history", 8))
        self.bg_alpha = float(pick("bg_alpha", 0.02))
        self.smooth_window = int(pick("smooth_window", 5))
        self.use_optical_flow = bool(pick("use_optical_flow", True))
        flow_cfg = smoke_cfg.get("optical_flow", cfg.get("optical_flow", {}))
        self.flow_cfg: Mapping[str, Any] = flow_cfg if isinstance(flow_cfg, Mapping) else {}

        self._bg_gray: Optional[np.ndarray] = None
        self._feature_history: Dict[int, Dict[str, Deque[float]]] = {}

        # Cache optical flow theo từng frame (tính 1 lần, dùng cho nhiều region)
        self._prev_gray: Optional[np.ndarray] = None
        self._flow: Optional[np.ndarray] = None
        self._flow_key: Optional[int] = None

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self._bg_gray = None
        self._feature_history.clear()
        self._prev_gray = None
        self._flow = None
        self._flow_key = None

    def prune(self, active_track_ids: Iterable[int]) -> None:
        """Xoá đệm đặc trưng của các track đã biến mất."""
        alive = set(int(t) for t in active_track_ids)
        for tid in list(self._feature_history.keys()):
            if tid not in alive:
                del self._feature_history[tid]

    # ------------------------------------------------------------------ #
    def _get_flow(self, bundle: Any) -> Optional[np.ndarray]:
        key = id(bundle.gray)
        if key == self._flow_key:
            return self._flow
        flow: Optional[np.ndarray] = None
        if (
            self.use_optical_flow
            and self._prev_gray is not None
            and self._prev_gray.shape == bundle.gray.shape
        ):
            try:
                from ..motion.optical_flow import compute_flow

                flow = compute_flow(self._prev_gray, bundle.gray, self.flow_cfg)
            except cv2.error:
                flow = None
        self._prev_gray = bundle.gray.copy()
        self._flow = flow
        self._flow_key = key
        return flow

    def update_background(
        self, gray: np.ndarray, update_mask: Optional[np.ndarray] = None
    ) -> None:
        """Cập nhật nền xám trung bình trượt; ``update_mask`` = 255 tại pixel được cập nhật."""
        gray_f = gray.astype(np.float32)
        if self._bg_gray is None or self._bg_gray.shape != gray_f.shape:
            self._bg_gray = gray_f.copy()
        else:
            cv2.accumulateWeighted(gray_f, self._bg_gray, self.bg_alpha, mask=update_mask)

    def _smoothed(self, track_id: int, name: str, value: Optional[float]) -> Optional[float]:
        """Trung bình trượt của một đặc trưng theo track."""
        if value is None:
            return None
        store = self._feature_history.setdefault(track_id, {})
        buf = store.setdefault(name, deque(maxlen=max(1, self.smooth_window)))
        buf.append(float(value))
        return float(np.mean(buf))

    # ------------------------------------------------------------------ #
    def update(self, region: Region, bundle: Any) -> Dict[str, float]:
        """Tính và ghi các đặc trưng khói vào ``region.extras``."""
        H, W = bundle.gray.shape[:2]
        x, y, w, h = region.bbox
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        if x2 <= x1 or y2 <= y1:
            return {}

        roi_mask = region.mask
        if roi_mask is not None and roi_mask.shape != (y2 - y1, x2 - x1):
            roi_mask = roi_mask[: (y2 - y1), : (x2 - x1)]

        # Đóng băng nền bên trong vùng khói ứng viên
        update_mask = np.full((H, W), 255, dtype=np.uint8)
        if roi_mask is not None:
            local = cv2.dilate(
                roi_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            )
            update_mask[y1:y2, x1:x2][local > 0] = 0
        self.update_background(bundle.gray, update_mask)
        assert self._bg_gray is not None

        curr_roi = bundle.gray[y1:y2, x1:x2]
        bg_roi = self._bg_gray[y1:y2, x1:x2].astype(np.uint8)

        # Chỉ đo gradient/LBP ở phần lõi mask để tránh biên khói sắc nét
        interior_mask: Optional[np.ndarray] = None
        if roi_mask is not None:
            interior_mask = cv2.erode(
                roi_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            )
            if np.count_nonzero(interior_mask) < 16:
                interior_mask = roi_mask

        # 1. edge_blur = energy_curr / energy_bg (Sobel)
        grad_curr = cv2.magnitude(
            cv2.Sobel(curr_roi, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(curr_roi, cv2.CV_32F, 0, 1, ksize=3),
        )
        grad_bg = cv2.magnitude(
            cv2.Sobel(bg_roi, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(bg_roi, cv2.CV_32F, 0, 1, ksize=3),
        )
        if interior_mask is not None and np.count_nonzero(interior_mask) > 0:
            valid = interior_mask > 0
            e_curr = float(np.mean(grad_curr[valid]))
            e_bg = float(np.mean(grad_bg[valid])) + 1e-6
        else:
            e_curr = float(np.mean(grad_curr))
            e_bg = float(np.mean(grad_bg)) + 1e-6
        edge_blur_ratio = float(np.clip(e_curr / e_bg, 0.0, 1.5))

        # 2. texture_smoothness = 1 - entropy chuẩn hoá
        entropy = _fast_uniform_lbp_entropy(curr_roi, interior_mask)
        texture_smoothness = float(np.clip(1.0 - (entropy - 2.0) / 5.5, 0.0, 1.0))

        # 3. upward_motion: Optical Flow (TV3) nếu có, fallback theo trọng tâm
        upward_motion: Optional[float] = None
        flow = self._get_flow(bundle)
        if flow is not None:
            from ..motion.optical_flow import compute_upward_motion

            upward_motion = compute_upward_motion(region, flow)
        # Khói mờ/ít texture có thể cho Optical Flow ~0 dù vẫn bốc lên:
        # fallback theo dịch chuyển trọng tâm khi flow không đủ tín hiệu.
        if (upward_motion is None or upward_motion <= 0.05) and len(region.history) >= 2:
            prev = region.history[-2]
            prev_cy = float(prev.get("centroid", region.centroid)[1])
            curr_cy = float(region.centroid[1])
            centroid_up = max(0.0, prev_cy - curr_cy)
            upward_motion = max(centroid_up, upward_motion or 0.0) if upward_motion is not None else centroid_up

        # 4. area_growth theo snapshot trước
        area_growth: Optional[float] = None
        if len(region.history) >= 2:
            prev = region.history[-2]
            prev_area = float(prev.get("area", region.area)) + 1e-6
            area_growth = (float(region.area) - prev_area) / prev_area

        features: Dict[str, float] = {}
        edge_smoothed = self._smoothed(region.track_id, "edge_blur", edge_blur_ratio)
        texture_smoothed = self._smoothed(
            region.track_id, "texture_smoothness", texture_smoothness
        )
        if edge_smoothed is not None:
            features["edge_blur"] = round(edge_smoothed, 4)
        if texture_smoothed is not None:
            features["texture_smoothness"] = round(texture_smoothed, 4)
        up_smoothed = self._smoothed(region.track_id, "upward_motion", upward_motion)
        if up_smoothed is not None:
            features["upward_motion"] = round(up_smoothed, 4)
        growth_smoothed = self._smoothed(region.track_id, "area_growth", area_growth)
        if growth_smoothed is not None:
            features["area_growth"] = round(growth_smoothed, 4)

        region.extras.update(features)
        return features
