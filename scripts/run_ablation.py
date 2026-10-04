"""Ablation study: bật dần từng tầng của pipeline và so sánh kết quả.

Ví dụ:
    python scripts/run_ablation.py --videos data/samples --annotations data/annotations
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402
from src.detectors import ClassicalDetector  # noqa: E402
from src.evaluation.evaluate import evaluate_dataset, summarize  # noqa: E402

VARIANTS: dict[str, dict] = {
    "1. Chỉ màu": dict(use_motion=False, use_temporal=False, use_persistence=False),
    "2. Màu + chuyển động": dict(use_motion=True, use_temporal=False, use_persistence=False),
    "3. + Bỏ phiếu thời gian": dict(use_motion=True, use_temporal=False, use_persistence=True),
    "4. + Đặc trưng thời gian (đầy đủ)": dict(use_motion=True, use_temporal=True,
                                              use_persistence=True),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--videos", default=str(ROOT / "data" / "samples"))
    p.add_argument("--annotations", default=str(ROOT / "data" / "annotations"))
    p.add_argument("--config", default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--out", default=str(ROOT / "outputs" / "ablation.csv"))
    args = p.parse_args()

    tables = []
    for name, flags in VARIANTS.items():
        print(f"\n=== {name} ===")
        cfg = load_config(args.config, overrides={"pipeline": flags})
        df = evaluate_dataset(lambda c=cfg: ClassicalDetector(c), args.videos,
                              args.annotations, args.max_frames)
        if df.empty:
            print("Không có video nào có nhãn.")
            return
        s = summarize(df)
        s.insert(0, "variant", name)
        tables.append(s)

    result = pd.concat(tables, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False, encoding="utf-8-sig")
    cols = ["variant", "label", "precision", "recall", "f1", "far", "mean_latency_s"]
    print("\n" + result[cols].round(3).to_string(index=False))
    print(f"\nĐã lưu: {args.out}")


if __name__ == "__main__":
    main()
