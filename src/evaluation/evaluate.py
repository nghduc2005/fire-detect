"""Chạy bộ phát hiện trên tập video có nhãn và tổng hợp kết quả.

Định dạng nhãn: `data/annotations/<tên_video_không_đuôi>.json`
    {
        "fire":  [[start_frame, end_frame], ...],
        "smoke": [[start_frame, end_frame], ...]
    }
Video "âm tính" (chỉ chứa vật gây nhiễu) dùng danh sách rỗng.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.detectors.base import BaseDetector
from src.structures import FIRE, SMOKE
from src.video_io import VideoReader

from .metrics import BinaryCounts, detection_latency, frame_level_counts, intervals_to_labels

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
LABELS = (FIRE, SMOKE)


def run_on_video(detector: BaseDetector, video_path: str | Path,
                 max_frames: int | None = None) -> tuple[dict[str, np.ndarray], float]:
    """Trả về ({nhãn: mảng bool theo frame}, fps)."""
    detector.reset()
    preds: dict[str, list[bool]] = {lb: [] for lb in LABELS}
    with VideoReader(video_path, max_frames=max_frames) as reader:
        fps = reader.fps
        if hasattr(detector, "fps"):
            detector.fps = fps
        for _, frame in reader:
            found = {d.label for d in detector.process(frame)}
            for lb in LABELS:
                preds[lb].append(lb in found)
    return {lb: np.asarray(v, bool) for lb, v in preds.items()}, fps


def evaluate_dataset(detector_factory: Callable[[], BaseDetector], video_dir: str | Path,
                     ann_dir: str | Path, max_frames: int | None = None) -> pd.DataFrame:
    """Đánh giá mọi video có file nhãn. Trả về DataFrame mỗi dòng = (video, nhãn)."""
    video_dir, ann_dir = Path(video_dir), Path(ann_dir)
    videos = sorted(p for p in video_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    rows = []
    detector = detector_factory()
    for vp in tqdm(videos, desc="Evaluating"):
        ann_path = ann_dir / f"{vp.stem}.json"
        if not ann_path.exists():
            print(f"[bỏ qua] {vp.name}: không có nhãn {ann_path.name}")
            continue
        ann = json.loads(ann_path.read_text(encoding="utf-8"))
        preds, fps = run_on_video(detector, vp, max_frames)
        for lb in LABELS:
            gt = intervals_to_labels(ann.get(lb, []), preds[lb].size)
            counts = frame_level_counts(preds[lb], gt)
            rows.append({"video": vp.name, "label": lb, **counts.as_dict(),
                         "latency_s": detection_latency(preds[lb], gt, fps)})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Gộp TP/FP/FN/TN theo nhãn rồi tính lại chỉ số (micro-average)."""
    out = []
    for lb, g in df.groupby("label"):
        c = BinaryCounts(int(g.tp.sum()), int(g.fp.sum()), int(g.fn.sum()), int(g.tn.sum()))
        out.append({"label": lb, **c.as_dict(), "mean_latency_s": g.latency_s.dropna().mean()})
    return pd.DataFrame(out)
