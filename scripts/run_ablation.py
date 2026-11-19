"""Thực nghiệm bóc tách (Ablation Study) — Thành viên 4.

Chạy 4 cấu hình pipeline bằng cách bật/tắt các cờ ``use_motion``,
``use_temporal``, ``use_persistence`` trong ``configs/default.yaml``::

    python -m scripts.run_ablation
    python -m scripts.run_ablation --data-dir data/raw --max-frames 120

Kết quả in ra bảng và lưu CSV tại ``outputs/ablation_results.csv``.
Quy ước nhãn video: tên file/thư mục chứa ``pos`` / ``fire`` / ``smoke`` là
positive; chứa ``neg`` / ``normal`` / ``non`` là negative (báo động giả).
"""

from __future__ import annotations

import argparse
import copy
import csv
import glob
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from src.config import DEFAULT_CONFIG_PATH, load_config, load_default_config
from src.detectors.classical import ClassicalDetector
from src.evaluation import DetectionEvaluator, evaluate_video

# 4 cấu hình bóc tách từ cơ bản đến đầy đủ
ABLATION_CONFIGS: List[Tuple[str, Dict[str, bool]]] = [
    (
        "1. Only Color (Baseline)",
        {"use_motion": False, "use_temporal": False, "use_persistence": False},
    ),
    (
        "2. Color + Motion",
        {"use_motion": True, "use_temporal": False, "use_persistence": False},
    ),
    (
        "3. Color + Motion + Temporal",
        {"use_motion": True, "use_temporal": True, "use_persistence": False},
    ),
    (
        "4. Full Pipeline (+ Persistence)",
        {"use_motion": True, "use_temporal": True, "use_persistence": True},
    ),
]

RESULT_COLUMNS: Sequence[str] = [
    "Accuracy (%)",
    "Precision (%)",
    "Recall (%)",
    "F1-Score (%)",
    "FPR (%)",
    "Mean IoU (%)",
    "FPS",
]


def collect_dataset(data_dir: str = "data/raw") -> List[Tuple[str, bool]]:
    """Quét video trong ``data_dir`` và suy ra nhãn positive/negative từ tên."""
    samples: List[Tuple[str, bool]] = []
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        for path in glob.glob(os.path.join(data_dir, "**", ext), recursive=True):
            lower_p = path.lower()
            if any(k in lower_p for k in ("neg", "normal", "non_fire", "non-fire")):
                samples.append((path, False))
            elif any(k in lower_p for k in ("pos", "fire", "smoke")):
                samples.append((path, True))
    return sorted(samples)


def run_suite(
    dataset: Sequence[Tuple[str, bool]],
    max_frames: int = 0,
    config_path: str | None = None,
    verbose: bool = False,
) -> List[Dict[str, object]]:
    """Chạy lần lượt 4 cấu hình trên dataset và trả về các dòng kết quả."""
    if config_path:
        base_cfg = load_config(config_path)
    else:
        base_cfg = load_default_config()

    results: List[Dict[str, object]] = []
    for name, flags in ABLATION_CONFIGS:
        cfg = copy.deepcopy(base_cfg)
        cfg["pipeline"].update(flags)
        detector = ClassicalDetector(cfg)
        evaluator = DetectionEvaluator()
        for video_path, is_pos in dataset:
            if verbose:
                tag = "POS" if is_pos else "NEG"
                print(f"  [{tag}] {video_path}")
            evaluate_video(
                detector,
                video_path,
                gt_positive=is_pos,
                evaluator=evaluator,
                max_frames=max_frames,
            )
        row: Dict[str, object] = {"Configuration": name}
        row.update(evaluator.summarize())
        results.append(row)
    return results


def print_table(results: Sequence[Dict[str, object]]) -> None:
    header = (
        f"{'Cấu hình Pipeline':<34} | {'Acc':>6} | {'Prec':>6} | {'Recall':>6} | "
        f"{'F1':>6} | {'FPR':>6} | {'mIoU':>6} | {'FPS':>5}"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{str(row['Configuration']):<34} | "
            f"{float(row['Accuracy (%)']):>5.1f}% | "
            f"{float(row['Precision (%)']):>5.1f}% | "
            f"{float(row['Recall (%)']):>5.1f}% | "
            f"{float(row['F1-Score (%)']):>5.1f}% | "
            f"{float(row['FPR (%)']):>5.1f}% | "
            f"{float(row['Mean IoU (%)']):>5.1f}% | "
            f"{float(row['FPS']):>5.1f}"
        )


def save_csv(results: Sequence[Dict[str, object]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Configuration", "TP", "FP", "TN", "FN", *RESULT_COLUMNS]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ablation Study cho pipeline lửa & khói")
    parser.add_argument("--data-dir", default="data/raw", help="Thư mục chứa video kiểm thử")
    parser.add_argument(
        "--max-frames", type=int, default=0, help="Giới hạn số frame mỗi video (0 = tất cả)"
    )
    parser.add_argument(
        "--config",
        default=None,
        help=f"File YAML cấu hình (mặc định: {DEFAULT_CONFIG_PATH.name})",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("outputs", "ablation_results.csv"),
        help="File CSV lưu kết quả",
    )
    parser.add_argument("--verbose", action="store_true", help="In từng video đang xử lý")
    args = parser.parse_args(argv)

    dataset = collect_dataset(args.data_dir)
    if not dataset:
        print(
            f"Không tìm thấy video trong {args.data_dir}.\n"
            "Hãy đặt video vào data/raw (tên chứa 'pos'/'fire'/'smoke' hoặc 'neg'/'normal')."
        )
        return

    print(f"Tìm thấy {len(dataset)} video kiểm thử.\n")
    results = run_suite(
        dataset,
        max_frames=args.max_frames,
        config_path=args.config,
        verbose=args.verbose,
    )
    print_table(results)
    save_csv(results, args.output)
    print(f"\nĐã lưu kết quả: {args.output}")


if __name__ == "__main__":
    main()
