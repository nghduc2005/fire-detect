"""Chạy pipeline trên một video / camera.

Ví dụ:
    python scripts/run_video.py --input data/samples/synthetic_fire.mp4 --show
    python scripts/run_video.py --input 0 --show --debug          # webcam
    python scripts/run_video.py --input x.mp4 --output outputs/x_result.mp4 --debug
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from src.config import load_config  # noqa: E402
from src.detectors import ClassicalDetector  # noqa: E402
from src.video_io import VideoReader, VideoWriter  # noqa: E402
from src.visualization import draw_detections, make_debug_grid  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phát hiện khói / lửa từ video")
    p.add_argument("--input", required=True, help="Đường dẫn video hoặc chỉ số camera (0, 1...)")
    p.add_argument("--config", default=None, help="File YAML ghi đè configs/default.yaml")
    p.add_argument("--output", default=None, help="Ghi video kết quả ra file")
    p.add_argument("--show", action="store_true", help="Hiển thị cửa sổ (nhấn q để thoát)")
    p.add_argument("--debug", action="store_true", help="Hiển thị/ghi lưới mask trung gian")
    p.add_argument("--max-frames", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    with VideoReader(args.input, max_frames=args.max_frames) as reader:
        detector = ClassicalDetector(cfg, fps=reader.fps)
        writer = VideoWriter(args.output, fps=reader.fps) if args.output else None
        n_alert, t0, n = 0, time.perf_counter(), 0
        for idx, frame in reader:
            detections = detector.process(frame)
            n_alert += bool(detections)
            vis = draw_detections(frame, detections)
            if args.debug:
                vis = make_debug_grid(vis, detector.debug, cfg["visualization"]["tile_width"])
            if writer:
                writer.write(vis)
            if args.show:
                cv2.imshow("Fire & Smoke Detection", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            n = idx + 1

        elapsed = time.perf_counter() - t0
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    print(f"Đã xử lý {n} frame trong {elapsed:.1f}s ({n / max(elapsed, 1e-6):.1f} FPS)")
    print(f"Số frame có cảnh báo: {n_alert}")
    if args.output:
        print(f"Đã ghi kết quả: {args.output}")


if __name__ == "__main__":
    main()
