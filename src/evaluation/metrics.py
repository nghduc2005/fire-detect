"""Chỉ số đánh giá mức frame và mức sự kiện."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class BinaryCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def __add__(self, other: "BinaryCounts") -> "BinaryCounts":
        return BinaryCounts(self.tp + other.tp, self.fp + other.fp,
                            self.fn + other.fn, self.tn + other.tn)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def false_alarm_rate(self) -> float:
        """Tỉ lệ frame âm tính bị báo nhầm: FP / (FP + TN)."""
        return self.fp / (self.fp + self.tn) if (self.fp + self.tn) else 0.0

    def as_dict(self) -> dict[str, float]:
        return {**asdict(self), "precision": self.precision, "recall": self.recall,
                "f1": self.f1, "far": self.false_alarm_rate}


def intervals_to_labels(intervals: Sequence[Sequence[int]], n_frames: int) -> np.ndarray:
    """[[start, end], ...] (bao gồm cả end) -> mảng bool độ dài n_frames."""
    labels = np.zeros(n_frames, dtype=bool)
    for start, end in intervals:
        labels[max(0, start):min(n_frames, end + 1)] = True
    return labels


def frame_level_counts(pred: np.ndarray, gt: np.ndarray) -> BinaryCounts:
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    n = min(pred.size, gt.size)
    pred, gt = pred[:n], gt[:n]
    return BinaryCounts(
        tp=int(np.sum(pred & gt)), fp=int(np.sum(pred & ~gt)),
        fn=int(np.sum(~pred & gt)), tn=int(np.sum(~pred & ~gt)),
    )


def detection_latency(pred: np.ndarray, gt: np.ndarray, fps: float) -> float | None:
    """Số giây từ frame đầu tiên có lửa/khói (GT) đến lần phát hiện đúng đầu tiên.
    None nếu video không có GT dương tính hoặc không bao giờ phát hiện được."""
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    n = min(pred.size, gt.size)
    pred, gt = pred[:n], gt[:n]
    gt_idx = np.flatnonzero(gt)
    if gt_idx.size == 0:
        return None
    hits = np.flatnonzero(pred[gt_idx[0]:] & gt[gt_idx[0]:])
    return float(hits[0] / fps) if hits.size else None
