"""
CLI chạy pipeline phát hiện khói / lửa trên một video.

Ví dụ::

    python scripts/run_video.py \\
        --input data/samples/fire.mp4 \\
        --output outputs/fire_detected.mp4 \\
        --config configs/default.yaml \\
        --show

Hoặc chạy từ camera webcam::

    python scripts/run_video.py --input 0 --show

Flags chính:

- ``--input``  : đường dẫn file video hoặc chỉ số camera (``0``, ``1``, ...).
- ``--output`` : đường dẫn file kết quả (``.mp4``). Để trống thì không ghi file.
- ``--config`` : file YAML cấu hình (mặc định: ``configs/default.yaml``).
- ``--show``   : hiển thị cửa sổ OpenCV preview (bấm ``q`` để thoát).
- ``--debug``  : vẽ thêm lưới mask trung gian (dùng cho báo cáo).
- ``--max-frames`` : giới hạn số frame xử lý (-1 = toàn bộ).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Cho phép ``python scripts/run_video.py`` chạy khi gốc dự án chưa trong PYTHONPATH.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2  # noqa: E402

from src.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from src.detectors import ClassicalDetector  # noqa: E402
from src.video_io import VideoReader, VideoWriter  # noqa: E402
from src.visualization import draw_detections, draw_fps_overlay, make_debug_grid  # noqa: E402


# ---------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chạy pipeline phát hiện khói / lửa trên 1 video.",
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="File video hoặc chỉ số camera (ví dụ: '0' cho webcam).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="File kết quả (.mp4). Để trống thì không ghi.",
    )
    parser.add_argument(
        "--config",
        "-c",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"File cấu hình YAML (mặc định: {DEFAULT_CONFIG_PATH.name}).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Hiển thị cửa sổ preview realtime (bấm q để thoát).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Vẽ thêm lưới mask trung gian trong preview / output.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="Giới hạn số frame xử lý (-1 = toàn bộ).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------- #
def _parse_source(raw: str):
    """Nếu ``raw`` chỉ gồm chữ số → camera index (int); ngược lại coi là path."""
    return int(raw) if raw.lstrip("-").isdigit() else raw


# ---------------------------------------------------------------------- #
def main(argv=None) -> int:
    args = parse_args(argv)

    config_path = Path(args.config)
    cfg = load_config(config_path)
    detector = ClassicalDetector(cfg)

    source = _parse_source(args.input)
    reader = VideoReader(source)
    writer = (
        VideoWriter(args.output, fps=reader.fps) if args.output else None
    )

    print(
        f"[run_video] source={source}  "
        f"fps={reader.fps:.1f}  size={reader.size}  "
        f"frames={reader.frame_count}  config={config_path.name}"
    )

    n_frames = 0
    n_dets_total = 0
    t_start = time.perf_counter()
    fps_running = 0.0
    try:
        for raw_frame in reader:
            t0 = time.perf_counter()
            detections = detector.process(raw_frame)
            dt = time.perf_counter() - t0
            fps_running = (0.9 * fps_running + 0.1 * (1.0 / max(dt, 1e-6))) if n_frames else 1.0 / max(dt, 1e-6)

            # Vẽ kết quả lên frame đã resize (lấy từ debug_masks shape).
            # Để trực quan, dựng lại frame hiển thị từ frame đã preprocess + bbox.
            bundle = detector.preprocess_only(raw_frame)  # (same resize như trong pipeline)
            display = draw_detections(bundle.bgr, detections)
            display = draw_fps_overlay(display, fps_running, len(detections))

            if args.debug and detector.debug_masks:
                display = make_debug_grid(display, detector.debug_masks)

            if writer is not None:
                writer.write(display)
            if args.show:
                cv2.imshow("fire-detect", display)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    print("[run_video] Người dùng dừng (q).")
                    break

            n_frames += 1
            n_dets_total += len(detections)
            if 0 < args.max_frames <= n_frames:
                break
    finally:
        reader.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    total = time.perf_counter() - t_start
    avg_fps = n_frames / total if total > 0 else 0.0
    print(
        f"[run_video] Xong: {n_frames} frames, {n_dets_total} detections, "
        f"{total:.2f}s, avg {avg_fps:.1f} fps"
    )
    if writer is not None:
        print(f"[run_video] Đã ghi: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
