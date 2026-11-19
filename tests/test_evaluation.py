"""Unit test cho module đánh giá chỉ số của Thành viên 4."""

from __future__ import annotations

from typing import List

import numpy as np

from src.detectors.base import BaseDetector, Detection
from src.evaluation import DetectionEvaluator, compute_iou, evaluate_video


def test_compute_iou() -> None:
    assert compute_iou((0, 0, 10, 10), (0, 0, 10, 10)) > 0.999
    assert compute_iou((0, 0, 10, 10), (10, 0, 10, 10)) < 1e-6
    assert abs(compute_iou((0, 0, 10, 10), (5, 0, 10, 10)) - 50 / 150) < 1e-6


def test_evaluator_confusion_matrix_and_metrics() -> None:
    ev = DetectionEvaluator()
    det = [Detection(bbox=(0, 0, 5, 5), label="fire", score=0.9)]
    ev.update_frame(det, gt_positive=True, elapsed_sec=0.01)   # TP
    ev.update_frame(det, gt_positive=False, elapsed_sec=0.01)  # FP
    ev.update_frame([], gt_positive=False, elapsed_sec=0.01)   # TN
    ev.update_frame([], gt_positive=True, elapsed_sec=0.01)    # FN

    res = ev.summarize()
    assert (res["TP"], res["FP"], res["TN"], res["FN"]) == (1, 1, 1, 1)
    assert res["Accuracy (%)"] == 50.0
    assert res["Precision (%)"] == 50.0
    assert res["Recall (%)"] == 50.0
    assert res["F1-Score (%)"] == 50.0
    assert res["FPR (%)"] == 50.0
    assert res["FPS"] == 100.0


def test_evaluator_mean_iou_and_reset() -> None:
    ev = DetectionEvaluator()
    det = [Detection(bbox=(0, 0, 10, 10), label="fire", score=0.9)]
    ev.update_frame(det, gt_positive=True, gt_boxes=[(0, 0, 10, 10), (50, 50, 10, 10)])
    assert ev.summarize()["Mean IoU (%)"] == 100.0

    ev.reset()
    assert ev.summarize()["TP"] == 0


class _EmptyDetector(BaseDetector):
    """Detector giả: không phát hiện gì, dùng để test vòng lặp video."""

    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def process(self, frame: np.ndarray) -> List[Detection]:
        return []


def test_evaluate_video(tmp_video_path) -> None:
    detector = _EmptyDetector()
    ev = DetectionEvaluator()
    evaluate_video(detector, str(tmp_video_path), gt_positive=True, evaluator=ev, max_frames=3)

    res = ev.summarize()
    assert detector.reset_count == 1
    assert res["FN"] == 3
    assert res["FPS"] > 0
    assert ev.report().startswith("Kết quả đánh giá")
