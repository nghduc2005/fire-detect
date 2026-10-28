"""Inspect image/video colour distributions to tune TV2 thresholds.

Example: ``python scripts/analyze_color.py --input data/raw/clip.mp4 --output outputs/colour.png``
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    # Windows terminals can default to CP1252, which cannot print Vietnamese
    # command-line help and error messages.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phân tích phân bố màu ảnh/video cho TV2.")
    parser.add_argument("--input", "-i", required=True, help="Đường dẫn ảnh hoặc video.")
    parser.add_argument("--max-frames", type=int, default=100, help="Số frame tối đa cần lấy mẫu.")
    parser.add_argument("--stride", type=int, default=10, help="Lấy một frame trên mỗi N frame.")
    parser.add_argument("--max-pixels", type=int, default=300_000, help="Giới hạn pixel dùng để vẽ.")
    parser.add_argument("--annotation", type=Path, default=None, help="CSV/JSON bbox; chỉ phân tích pixel thuộc nhãn.")
    parser.add_argument("--label", choices=["fire", "smoke"], default=None, help="Lọc một nhãn trong annotation.")
    parser.add_argument("--output", "-o", type=Path, default=None, help="PNG đầu ra; bỏ trống để hiển thị.")
    return parser.parse_args(argv)


def _record_bbox(record: dict[str, Any]) -> tuple[int, int, int, int]:
    """Read a bbox in either ``bbox: [x, y, w, h]`` or flat CSV form."""
    bbox = record.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return tuple(int(float(value)) for value in bbox)  # type: ignore[return-value]
    try:
        return tuple(int(float(record[key])) for key in ("x", "y", "width", "height"))  # type: ignore[return-value]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Mỗi annotation cần bbox [x, y, w, h] hoặc x,y,width,height.") from exc


def load_annotations(path: Path, label: str | None) -> dict[int, list[tuple[int, int, int, int]]]:
    """Load frame-indexed bboxes from simple CSV or JSON annotations.

    CSV columns: ``frame_index,label,x,y,width,height``. JSON can be a list
    of the same records or an object whose ``annotations``/``frames`` field is
    that list. For a still image, use ``frame_index: 0``.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy annotation: {path}")
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            records: Any = list(csv.DictReader(handle))
    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            loaded: Any = json.load(handle)
        records = loaded.get("annotations", loaded.get("frames", [])) if isinstance(loaded, dict) else loaded
    else:
        raise ValueError("Annotation chỉ hỗ trợ .csv hoặc .json")
    if not isinstance(records, list):
        raise ValueError("Annotation phải là danh sách các record theo frame.")

    bboxes: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Mỗi annotation phải là một object/dòng CSV hợp lệ.")
        record_label = str(record.get("label", "")).lower()
        if label is not None and record_label != label:
            continue
        try:
            frame_index = int(record.get("frame_index", record.get("frame", 0)))
        except (TypeError, ValueError) as exc:
            raise ValueError("frame_index phải là số nguyên.") from exc
        bboxes[frame_index].append(_record_bbox(record))
    if not bboxes:
        target = f" cho nhãn '{label}'" if label else ""
        raise ValueError(f"Không tìm thấy bounding box annotation{target}.")
    return dict(bboxes)


def _pixels_in_bboxes(frame: np.ndarray, bboxes: list[tuple[int, int, int, int]]) -> list[np.ndarray]:
    """Clip bboxes to a frame and return non-empty BGR crops as pixel arrays."""
    height, width = frame.shape[:2]
    crops: list[np.ndarray] = []
    for x, y, box_width, box_height in bboxes:
        left, top = max(0, x), max(0, y)
        right, bottom = min(width, x + box_width), min(height, y + box_height)
        if right > left and bottom > top:
            crops.append(frame[top:bottom, left:right].reshape(-1, 3))
    return crops


def read_samples(
    path: Path,
    max_frames: int,
    stride: int,
    annotations: dict[int, list[tuple[int, int, int, int]]] | None = None,
) -> np.ndarray:
    """Read all frame pixels, or annotated bbox pixels, from an image/video."""
    if path.suffix.lower() in IMAGE_SUFFIXES:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Không đọc được ảnh: {path}")
        if annotations is None:
            return image.reshape(-1, 3)
        crops = _pixels_in_bboxes(image, annotations.get(0, []))
        if not crops:
            raise ValueError("Không có bbox hợp lệ cho ảnh ở frame_index 0.")
        return np.concatenate(crops, axis=0)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Không mở được video: {path}")
    frames: list[np.ndarray] = []
    sampled_frames = 0
    last_annotation_frame = max(annotations) if annotations else None
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if annotations is None and index % stride == 0:
                frames.append(frame.reshape(-1, 3))
                sampled_frames += 1
                if sampled_frames >= max_frames:
                    break
            elif annotations is not None:
                if index in annotations:
                    crops = _pixels_in_bboxes(frame, annotations[index])
                    if crops:
                        frames.extend(crops)
                        sampled_frames += 1
                        if sampled_frames >= max_frames:
                            break
                if index >= last_annotation_frame:
                    break
            index += 1
    finally:
        capture.release()
    if not frames:
        raise ValueError("Video không có frame nào để phân tích.")
    return np.concatenate(frames, axis=0)


def plot_distributions(bgr_pixels: np.ndarray, max_pixels: int):
    """Create RGB/HSV/YCrCb histograms plus Cb-Cr and H-S scatter plots."""
    rng = np.random.default_rng(42)
    if len(bgr_pixels) > max_pixels:
        bgr_pixels = bgr_pixels[rng.choice(len(bgr_pixels), size=max_pixels, replace=False)]
    image = bgr_pixels.reshape(-1, 1, 3).astype(np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).reshape(-1, 3)
    rgb = bgr_pixels[:, ::-1]
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    for channel, colour, name in zip(range(3), ("red", "green", "blue"), ("R", "G", "B")):
        axes[0, 0].hist(rgb[:, channel], bins=64, range=(0, 256), color=colour, alpha=0.45, label=name)
    axes[0, 0].set(title="RGB histogram", xlabel="Value", ylabel="Pixels")
    axes[0, 0].legend()
    for channel, colour, name in zip(range(3), ("orange", "green", "black"), ("H", "S", "V")):
        axes[0, 1].hist(hsv[:, channel], bins=64, color=colour, alpha=0.45, label=name)
    axes[0, 1].set(title="HSV histogram", xlabel="Value", ylabel="Pixels")
    axes[0, 1].legend()
    for channel, colour, name in zip(range(3), ("black", "crimson", "royalblue"), ("Y", "Cr", "Cb")):
        axes[0, 2].hist(ycrcb[:, channel], bins=64, range=(0, 256), color=colour, alpha=0.45, label=name)
    axes[0, 2].set(title="YCrCb histogram", xlabel="Value", ylabel="Pixels")
    axes[0, 2].legend()
    axes[1, 0].scatter(ycrcb[:, 2], ycrcb[:, 1], s=1, alpha=0.08, c="darkorange")
    axes[1, 0].set(title="Cb-Cr distribution", xlabel="Cb", ylabel="Cr", xlim=(0, 255), ylim=(0, 255))
    axes[1, 1].scatter(hsv[:, 0], hsv[:, 1], s=1, alpha=0.08, c="slategray")
    axes[1, 1].set(title="Hue-Saturation distribution", xlabel="H", ylabel="S", xlim=(0, 180), ylim=(0, 255))
    axes[1, 2].axis("off")
    axes[1, 2].text(0.02, 0.95, f"Pixels analysed: {len(bgr_pixels):,}", va="top", fontsize=12)
    figure.tight_layout()
    return figure


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    path = Path(args.input)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy input: {path}")
    annotations = load_annotations(args.annotation, args.label) if args.annotation else None
    pixels = read_samples(path, max(1, args.max_frames), max(1, args.stride), annotations)
    figure = plot_distributions(pixels, max(1, args.max_pixels))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=160)
        print(f"Đã lưu biểu đồ: {args.output}")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
