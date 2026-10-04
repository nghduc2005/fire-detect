import cv2
import numpy as np

from src.config import load_config
from src.detectors import ClassicalDetector
from src.evaluation import detection_latency, frame_level_counts, intervals_to_labels
from src.visualization import draw_detections, make_debug_grid


def _flicker_fire_frames(n=60, seed=0):
    rng = np.random.default_rng(seed)
    for t in range(n):
        img = np.full((360, 640, 3), (40, 70, 40), np.uint8)
        if t >= 10:
            for _ in range(6):
                c = (320 + int(rng.integers(-15, 15)), 250 - int(rng.integers(0, 40)))
                ax = (int(rng.integers(8, 22)), int(rng.integers(20, 45)))
                cv2.ellipse(img, c, ax, 0, 0, 360,
                            (int(rng.integers(0, 40)), int(rng.integers(90, 200)), 255), -1)
        yield img


def test_pipeline_detects_flickering_fire():
    det = ClassicalDetector(load_config(), fps=25)
    labels = [{d.label for d in det.process(f)} for f in _flicker_fire_frames()]
    assert any("fire" in s for s in labels[30:]), "Phải phát hiện lửa nhấp nháy"
    assert not any(labels[:10]), "Không được báo khi chưa có lửa"


def test_pipeline_rejects_static_red_object():
    det = ClassicalDetector(load_config(), fps=25)
    img = np.full((360, 640, 3), (40, 70, 40), np.uint8)
    cv2.rectangle(img, (200, 150), (300, 250), (20, 140, 250), -1)
    results = [det.process(img.copy()) for _ in range(60)]
    assert not any(results[5:]), "Vật đỏ đứng yên phải bị loại bởi tầng chuyển động"


def test_debug_outputs_and_drawing():
    det = ClassicalDetector(load_config(), fps=25)
    frames = list(_flicker_fire_frames(20))
    dets = []
    for f in frames:
        dets = det.process(f)
    assert {"fire_color", "motion", "fire_candidates"} <= det.debug.keys()
    grid = make_debug_grid(draw_detections(frames[-1], dets), det.debug)
    assert grid.ndim == 3


def test_metrics():
    gt = intervals_to_labels([[2, 5]], 10)
    pred = np.zeros(10, bool)
    pred[4:8] = True
    c = frame_level_counts(pred, gt)
    assert (c.tp, c.fp, c.fn, c.tn) == (2, 2, 2, 4)
    assert detection_latency(pred, gt, fps=1.0) == 2.0
