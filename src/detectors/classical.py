"""Pipeline phát hiện khói / lửa bằng xử lý ảnh truyền thống.

    frame -> tiền xử lý -> mask màu (lửa, khói) ∩ mask chuyển động
          -> hình thái học -> vùng ứng viên -> lọc hình học
          -> đo tín hiệu từng frame -> tracking
          -> đặc trưng thời gian -> chấm điểm + bỏ phiếu -> Detection
"""
from __future__ import annotations

import cv2
import numpy as np

from src.color import fire_color_mask, smoke_color_mask
from src.filtering import clean_mask, extract_regions, is_persistent, passes_geometry, score_features
from src.filtering.candidates import region_mask
from src.motion import MotionDetector, OpticalFlowAnalyzer
from src.preprocessing import PreprocessedFrame, preprocess
from src.structures import FIRE, SMOKE, Detection, Region
from src.temporal import (PixelFlickerMap, area_growth, area_variation, boundary_roughness,
                          edge_blur_ratio, shape_change, temporal_flicker_score,
                          texture_smoothness, upward_velocity)
from src.tracking import RegionTracker, Track

from .base import BaseDetector


class ClassicalDetector(BaseDetector):
    def __init__(self, cfg: dict, fps: float = 25.0):
        super().__init__()
        self.cfg = cfg
        self.fps = fps
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self) -> None:
        cfg = self.cfg
        self.motion = MotionDetector(cfg["motion"])
        self.flow = OpticalFlowAnalyzer(cfg["optical_flow"])
        self.flicker_map = PixelFlickerMap(cfg["temporal"]["flicker_window"])
        self.tracker = RegionTracker(**cfg["tracking"])
        self.frame_idx = 0
        self.debug = {}
        self.tracks: list[Track] = []

    # ---------------------------------------------------------------- process
    def process(self, frame: np.ndarray) -> list[Detection]:
        cfg, pipe = self.cfg, self.cfg["pipeline"]
        pf = preprocess(frame, cfg["preprocessing"])
        h, w = pf.shape
        zeros = np.zeros((h, w), np.uint8)

        # 1) Chuyển động
        if pipe["use_motion"]:
            motion = self.motion.apply(pf.blurred, pf.gray)
        else:
            motion = np.full((h, w), 255, np.uint8)
        self.flow.set_frame(pf.gray)

        # 2) Màu
        fire_col = fire_color_mask(pf, cfg["fire_color"]) if pipe["detect_fire"] else zeros
        smoke_col = smoke_color_mask(pf, cfg["smoke_color"]) if pipe["detect_smoke"] else zeros
        flicker_counts = self.flicker_map.update(fire_col)

        # 3) Kết hợp + hình thái học
        m_cfg = cfg["morphology"]
        fire_cand = clean_mask(cv2.bitwise_and(fire_col, motion), **m_cfg["fire"])
        smoke_cand = clean_mask(cv2.bitwise_and(smoke_col, motion), **m_cfg["smoke"])
        # Pixel vừa là lửa vừa là khói -> ưu tiên lửa
        smoke_cand = cv2.bitwise_and(smoke_cand, cv2.bitwise_not(fire_cand))

        # 4) Vùng ứng viên + lọc hình học
        c_cfg = cfg["candidates"]
        max_area = c_cfg["max_area_ratio"] * h * w
        regions = (extract_regions(fire_cand, FIRE, c_cfg["min_area"], max_area)
                   + extract_regions(smoke_cand, SMOKE, c_cfg["min_area"], max_area))
        regions = [r for r in regions if passes_geometry(r, c_cfg)]

        # 5) Tín hiệu từng frame cho mỗi vùng
        for r in regions:
            self._measure_signals(r, pf, flicker_counts)

        # 6) Tracking + quyết định
        self.tracks = self.tracker.update(regions)
        detections = [d for t in self.tracks if (d := self._decide(t, pf)) is not None]

        self.debug = {
            "fire_color": fire_col,
            "smoke_color": smoke_col,
            "motion": motion,
            "flicker": cv2.convertScaleAbs(self.flicker_map.normalized(), alpha=255),
            "fire_candidates": fire_cand,
            "smoke_candidates": smoke_cand,
        }
        self.frame_idx += 1
        return detections

    # ------------------------------------------------------------- internals
    def _measure_signals(self, r: Region, pf: PreprocessedFrame, flicker_counts: np.ndarray) -> None:
        x, y, w, h = r.bbox
        m = region_mask(pf.shape, r) > 0
        if r.label == FIRE:
            window = max(1, self.flicker_map.window)
            r.signals["pixel_flicker"] = float(flicker_counts[y:y + h, x:x + w][m].mean()) / window
            r.signals["mean_intensity"] = float(pf.gray[y:y + h, x:x + w][m].mean())
        else:
            r.signals["edge_blur"] = edge_blur_ratio(pf.gray, self.motion.background_gray(), r.bbox)
            r.signals["texture_smoothness"] = texture_smoothness(pf.gray, r.bbox)
            motion = self.flow.region_motion(r.bbox)
            r.signals["flow_dy"] = motion["mean_dy"] if motion else None

    def _track_features(self, t: Track) -> dict[str, float | None]:
        hist = list(t.history)
        areas = [r.area for r in hist]

        def mean_signal(key: str) -> float | None:
            vals = [r.signals.get(key) for r in hist]
            vals = [v for v in vals if v is not None]
            return float(np.mean(vals)) if vals else None

        if t.label == FIRE:
            intensities = [r.signals.get("mean_intensity", 0.0) for r in hist]
            return {
                "pixel_flicker": mean_signal("pixel_flicker"),
                "intensity_flicker": temporal_flicker_score(
                    intensities, self.fps, self.cfg["temporal"]["flicker_min_freq"]),
                "area_variation": area_variation(areas),
                "shape_change": shape_change([r.contour for r in hist]),
                "boundary_roughness": float(np.mean([boundary_roughness(r.contour) for r in hist])),
            }

        flow_dy = mean_signal("flow_dy")
        upward = -flow_dy if flow_dy is not None else upward_velocity([r.centroid for r in hist])
        return {
            "edge_blur": mean_signal("edge_blur"),
            "upward_motion": upward,
            "area_growth": area_growth(areas),
            "texture_smoothness": mean_signal("texture_smoothness"),
        }

    def _decide(self, t: Track, pf: PreprocessedFrame) -> Detection | None:
        pipe = self.cfg["pipeline"]
        if not t.is_visible:
            return None

        features: dict[str, float | None] = {}
        score = 1.0
        if pipe["use_temporal"]:
            if len(t.history) < self.cfg["temporal"]["min_history"]:
                return None
            features = self._track_features(t)
            spec = self.cfg["scoring"][t.label]
            score = score_features(features, spec["features"])
            if score < spec["threshold"]:
                t.last_score = score
                return None
        t.last_score = score

        if pipe["use_persistence"] and not is_persistent(t, **self.cfg["persistence"]):
            return None

        s = pf.scale
        x, y, w, h = t.bbox
        return Detection(
            bbox=(int(x * s), int(y * s), int(w * s), int(h * s)),
            label=t.label, score=float(score), track_id=t.track_id,
            features={k: v for k, v in features.items() if v is not None},
        )
