"""Vẽ kết quả và ghép lưới ảnh trung gian (phục vụ demo + báo cáo)."""
from __future__ import annotations

import cv2
import numpy as np

from src.structures import FIRE, SMOKE, Detection

COLORS = {FIRE: (0, 80, 255), SMOKE: (220, 220, 220)}  # BGR


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    out = frame.copy()
    thickness = max(2, frame.shape[1] // 400)
    for d in detections:
        x, y, w, h = d.bbox
        color = COLORS.get(d.label, (0, 255, 0))
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
        text = f"{d.label.upper()} {d.score:.2f}" + (f" #{d.track_id}" if d.track_id else "")
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x, max(0, y - th - 6)), (x + tw + 4, y), color, cv2.FILLED)
        cv2.putText(out, text, (x + 2, max(th, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)
    if any(d.label == FIRE for d in detections) or any(d.label == SMOKE for d in detections):
        cv2.putText(out, "ALERT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
                    cv2.LINE_AA)
    return out


def _to_tile(img: np.ndarray, title: str, width: int) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    tile = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    cv2.rectangle(tile, (0, 0), (width, 22), (0, 0, 0), cv2.FILLED)
    cv2.putText(tile, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA)
    return tile


def make_debug_grid(frame: np.ndarray, debug: dict[str, np.ndarray],
                    tile_width: int = 320, cols: int = 4) -> np.ndarray:
    """Ghép frame kết quả + các mask trung gian thành một lưới ảnh."""
    tiles = [_to_tile(frame, "result", tile_width)]
    if "flicker" in debug:
        debug = {**debug, "flicker": cv2.applyColorMap(debug["flicker"], cv2.COLORMAP_INFERNO)}
    tiles += [_to_tile(img, name, tile_width) for name, img in debug.items()]
    th = tiles[0].shape[0]
    tiles = [cv2.resize(t, (tile_width, th)) if t.shape[0] != th else t for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)
