"""Đánh giá hiệu năng detector và thực nghiệm bóc tách — Thành viên 4.

Cung cấp:

- :func:`compute_iou` — IoU giữa 2 box ``(x, y, w, h)``.
- :class:`DetectionEvaluator` — tổng hợp TP/FP/TN/FN và tính
  Accuracy, Precision, Recall, F1-Score, FPR, Mean IoU, FPS.
- :func:`evaluate_video` — chạy detector trên 1 video và cập nhật evaluator.

Đánh giá ở cấp **frame** (video positive = có cháy/khói trong cả video);
IoU được tính khi có ground-truth box truyền vào ``gt_boxes``.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..detectors.base import BaseDetector, Detection

__all__ = ["DetectionEvaluator", "compute_iou", "evaluate_video"]


def compute_iou(
    box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]
) -> float:
    """Tính IoU giữa 2 box định dạng ``(x, y, w, h)``."""
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union_area = float(aw * ah + bw * bh - inter_area) + 1e-6
    return inter_area / union_area


class DetectionEvaluator:
    """Tổng hợp chỉ số Precision, Recall, F1, FPR, Accuracy, Mean IoU và FPS.

    Ví dụ::

        ev = DetectionEvaluator()
        ev.update_frame(detections, gt_positive=True, elapsed_sec=0.02)
        print(ev.summarize())
    """

    def __init__(self) -> None:
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.ious: List[float] = []
        self.frame_times: List[float] = []

    def reset(self) -> None:
        self.__init__()

    def update_frame(
        self,
        detections: Sequence[Detection],
        gt_positive: bool,
        gt_boxes: Optional[Sequence[Tuple[int, int, int, int]]] = None,
        elapsed_sec: float = 0.0,
    ) -> None:
        """Cập nhật confusion matrix cho 1 frame."""
        pred_positive = len(detections) > 0
        if pred_positive and gt_positive:
            self.tp += 1
        elif pred_positive and not gt_positive:
            self.fp += 1
        elif not pred_positive and not gt_positive:
            self.tn += 1
        else:
            self.fn += 1

        if elapsed_sec > 0:
            self.frame_times.append(float(elapsed_sec))

        if gt_boxes and detections:
            for det in detections:
                best_iou = max(compute_iou(det.bbox, gb) for gb in gt_boxes)
                self.ious.append(best_iou)

    def summarize(self) -> Dict[str, float]:
        """Trả về dict chỉ số (phần trăm cho các metric tỷ lệ)."""
        precision = self.tp / (self.tp + self.fp + 1e-9)
        recall = self.tp / (self.tp + self.fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        fpr = self.fp / (self.fp + self.tn + 1e-9)
        acc = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + 1e-9)
        mean_iou = float(np.mean(self.ious)) if self.ious else 0.0
        avg_fps = (1.0 / float(np.mean(self.frame_times))) if self.frame_times else 0.0
        return {
            "TP": self.tp,
            "FP": self.fp,
            "TN": self.tn,
            "FN": self.fn,
            "Accuracy (%)": round(acc * 100, 2),
            "Precision (%)": round(precision * 100, 2),
            "Recall (%)": round(recall * 100, 2),
            "F1-Score (%)": round(f1 * 100, 2),
            "FPR (%)": round(fpr * 100, 2),
            "Mean IoU (%)": round(mean_iou * 100, 2),
            "FPS": round(avg_fps, 1),
        }

    def report(self, title: str = "Kết quả đánh giá") -> str:
        """Báo cáo dạng text để in/log."""
        res = self.summarize()
        lines = [title, "-" * len(title)]
        for key, value in res.items():
            if isinstance(value, int):
                lines.append(f"{key:<15}: {value}")
            else:
                suffix = "" if key == "FPS" else "%"
                lines.append(f"{key:<15}: {value:>8.2f}{suffix}")
        return "\n".join(lines)


def evaluate_video(
    detector: BaseDetector,
    video_path: str,
    gt_positive: bool,
    evaluator: DetectionEvaluator,
    max_frames: int = 0,
) -> None:
    """Chạy detector trên 1 file video và cập nhật kết quả vào ``evaluator``.

    Parameters
    ----------
    detector : BaseDetector
        Detector đã khởi tạo (sẽ được ``reset()`` trước khi chạy).
    video_path : str
        Đường dẫn video.
    gt_positive : bool
        ``True`` nếu video có cháy/khói (nhãn cấp video).
    evaluator : DetectionEvaluator
        Bộ tích luỹ kết quả.
    max_frames : int
        Giới hạn số frame (0 = toàn bộ video).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Không mở được video: {video_path}")
    detector.reset()
    frame_idx = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if max_frames > 0 and frame_idx > max_frames:
                break
            t0 = time.perf_counter()
            detections = detector.process(frame)
            dt = time.perf_counter() - t0
            evaluator.update_frame(detections, gt_positive=gt_positive, elapsed_sec=dt)
    finally:
        cap.release()
