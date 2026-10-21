"""
``ClassicalDetector`` — tích hợp pipeline XLA truyền thống.

Thứ tự tầng lọc (xem README mục 1):

1. **Tiền xử lý**                 : :class:`~src.preprocessing.Preprocessor`.
2. **Phân tích màu lửa + khói**   : :mod:`src.color` (TV2).
3. **Phát hiện chuyển động**      : :class:`~src.motion.MotionDetector` (TV3).
4. **AND mask màu ∩ chuyển động**.
5. **Hình thái học + candidates** : :mod:`src.filtering` (TV4).
6. **Theo dõi vùng**              : :class:`~src.tracking.RegionTracker` (TV3).
7. **Đặc trưng thời gian**        : :mod:`src.temporal` (TV3 + TV4).
8. **Chấm điểm + Persistence**    : :class:`~src.filtering.FeatureScorer`
   và :class:`~src.filtering.PersistenceVoter` (TV4).

Mỗi tầng có thể bật / tắt qua ``configs/default.yaml`` → ``pipeline.*`` để
phục vụ **ablation study** của TV4.

Pipeline **chạy end-to-end ngay cả khi module của TV2/3/4 chỉ là stub** —
đúng tinh thần "MVP end-to-end sớm" của kế hoạch (tuần 3).
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Tuple

import numpy as np

from ..color import fire_color_mask, smoke_color_mask
from ..config import ConfigDict, load_default_config
from ..filtering import (
    FeatureScorer,
    PersistenceVoter,
    extract_candidates,
    morphology_clean,
)
from ..motion import MotionDetector
from ..preprocessing import Preprocessor
from ..temporal import FireTemporalAnalyzer, SmokeTemporalAnalyzer
from ..tracking import RegionTracker
from .base import BaseDetector, Detection, Region

__all__ = ["ClassicalDetector"]


class ClassicalDetector(BaseDetector):
    """Bộ phát hiện khói / lửa dùng xử lý ảnh truyền thống."""

    # ------------------------------------------------------------------ #
    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        # Chuẩn hóa config về ConfigDict để truy cập cả attribute lẫn mapping.
        if config is None:
            self.cfg: ConfigDict = load_default_config()
        elif isinstance(config, ConfigDict):
            self.cfg = config
        else:
            self.cfg = ConfigDict(dict(config))

        pl = self.cfg.get("pipeline", {})
        self.detect_fire: bool = bool(pl.get("detect_fire", True))
        self.detect_smoke: bool = bool(pl.get("detect_smoke", True))
        self.use_motion: bool = bool(pl.get("use_motion", True))
        self.use_temporal: bool = bool(pl.get("use_temporal", True))
        self.use_persistence: bool = bool(pl.get("use_persistence", True))

        # --- Các thành phần của pipeline ------------------------------- #
        self.preprocessor = Preprocessor(self.cfg)
        self.motion = MotionDetector(self.cfg.get("motion", {}))

        self.fire_tracker = RegionTracker(self.cfg.get("tracking", {}))
        self.smoke_tracker = RegionTracker(self.cfg.get("tracking", {}))

        self.fire_temporal = FireTemporalAnalyzer(self.cfg.get("temporal", {}))
        self.smoke_temporal = SmokeTemporalAnalyzer(self.cfg.get("temporal", {}))

        scoring = self.cfg.get("scoring", {})
        self.fire_scorer = FeatureScorer(scoring.get("fire", {}), kind="fire")
        self.smoke_scorer = FeatureScorer(scoring.get("smoke", {}), kind="smoke")

        self.fire_voter = PersistenceVoter(self.cfg.get("persistence", {}))
        self.smoke_voter = PersistenceVoter(self.cfg.get("persistence", {}))

        # Lưu lại cấu hình morphology / candidates cho tiện gọi.
        morph = self.cfg.get("morphology", {})
        self._morph_fire = morph.get("fire", {"open_ksize": 3, "close_ksize": 7})
        self._morph_smoke = morph.get("smoke", {"open_ksize": 5, "close_ksize": 11})
        self._candidates_cfg = self.cfg.get("candidates", {})

        # Mask debug phục vụ app Streamlit (populate ở mỗi lần process).
        self._debug_masks: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ #
    @property
    def debug_masks(self) -> dict[str, np.ndarray]:
        """Mask trung gian của frame **vừa xử lý xong**. App Streamlit đọc để hiển thị."""
        return self._debug_masks

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Reset mọi thành phần có trạng thái (gọi khi bắt đầu video mới)."""
        self.motion.reset()
        self.fire_tracker.reset()
        self.smoke_tracker.reset()
        self.fire_temporal.reset()
        self.smoke_temporal.reset()
        self._debug_masks = {}

    # ------------------------------------------------------------------ #
    def process(self, frame: np.ndarray) -> List[Detection]:
        """Chạy toàn bộ pipeline trên 1 frame BGR, trả danh sách phát hiện."""
        # 1. Tiền xử lý — chuẩn hóa kích thước + chuyển không gian màu.
        bundle = self.preprocessor(frame)

        # 2. Mask màu
        fire_mask = (
            fire_color_mask(bundle, self.cfg.get("fire_color", {}))
            if self.detect_fire
            else np.zeros((bundle.height, bundle.width), dtype=np.uint8)
        )
        smoke_mask = (
            smoke_color_mask(bundle, self.cfg.get("smoke_color", {}))
            if self.detect_smoke
            else np.zeros((bundle.height, bundle.width), dtype=np.uint8)
        )

        # 3. Mask chuyển động + AND
        if self.use_motion:
            motion_mask = self.motion.update(bundle)
            if self.detect_fire:
                fire_mask = cv2_and(fire_mask, motion_mask)
            if self.detect_smoke:
                smoke_mask = cv2_and(smoke_mask, motion_mask)
        else:
            motion_mask = np.zeros((bundle.height, bundle.width), dtype=np.uint8)

        # 4. Morphology
        fire_clean = morphology_clean(
            fire_mask,
            open_ksize=int(self._morph_fire.get("open_ksize", 3)),
            close_ksize=int(self._morph_fire.get("close_ksize", 7)),
        )
        smoke_clean = morphology_clean(
            smoke_mask,
            open_ksize=int(self._morph_smoke.get("open_ksize", 5)),
            close_ksize=int(self._morph_smoke.get("close_ksize", 11)),
        )

        # 5. Trích xuất candidates
        fire_regions = (
            extract_candidates(fire_clean, label="fire", cfg=self._candidates_cfg)
            if self.detect_fire
            else []
        )
        smoke_regions = (
            extract_candidates(smoke_clean, label="smoke", cfg=self._candidates_cfg)
            if self.detect_smoke
            else []
        )

        # 6. Tracking
        fire_regions = self.fire_tracker.update(fire_regions)
        smoke_regions = self.smoke_tracker.update(smoke_regions)

        # 7. Đặc trưng thời gian + Scoring
        detections: List[Detection] = []
        if self.detect_fire:
            detections.extend(
                self._score_regions(
                    fire_regions,
                    bundle,
                    self.fire_temporal,
                    self.fire_scorer,
                    self.fire_voter,
                )
            )
        if self.detect_smoke:
            detections.extend(
                self._score_regions(
                    smoke_regions,
                    bundle,
                    self.smoke_temporal,
                    self.smoke_scorer,
                    self.smoke_voter,
                )
            )

        # 8. Lưu mask debug (app Streamlit).
        self._debug_masks = {
            "fire_color": fire_mask,
            "smoke_color": smoke_mask,
            "motion": motion_mask,
            "fire_clean": fire_clean,
            "smoke_clean": smoke_clean,
        }

        return detections

    # ------------------------------------------------------------------ #
    def _score_regions(
        self,
        regions: List[Region],
        bundle,
        temporal_analyzer,
        scorer: FeatureScorer,
        voter: PersistenceVoter,
    ) -> List[Detection]:
        """Chạy temporal → scoring → persistence trên 1 nhóm region."""
        results: List[Detection] = []
        for region in regions:
            if self.use_temporal:
                temporal_analyzer.update(region, bundle)
            score, breakdown = scorer.score(region)
            if score < scorer.threshold:
                continue
            if self.use_persistence and not voter.vote(region):
                continue
            results.append(
                Detection(
                    bbox=region.bbox,
                    label=region.label,
                    score=float(score),
                    track_id=int(region.track_id),
                    extras={"breakdown": breakdown, **region.extras},
                )
            )
        return results

    # ------------------------------------------------------------------ #
    def preprocess_only(self, frame: np.ndarray):
        """Shortcut: trả về :class:`FrameBundle` sau tiền xử lý — tiện debug."""
        return self.preprocessor(frame)


# ---------------------------------------------------------------------- #
# Helper nội bộ: AND an toàn giữa 2 mask uint8 (nếu kích thước khớp).
# ---------------------------------------------------------------------- #
def cv2_and(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Phép ``and`` theo bit, có kiểm tra kích thước & kiểu dữ liệu."""
    if a.shape != b.shape:
        raise ValueError(
            f"Mask khác kích thước: {a.shape} vs {b.shape}"
        )
    # Nhân thẳng là OK vì mask uint8 {0, 255}; dùng cv2.bitwise_and để chắc chắn.
    import cv2  # noqa: WPS433 — tránh overhead khi import module
    return cv2.bitwise_and(a, b)
