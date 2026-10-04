"""Chấm điểm tổng hợp từ các đặc trưng (weighted sum của các hàm ramp).

Mỗi đặc trưng có cấu hình [trọng_số, lo, hi]:
    điểm_thành_phần = clip((x - lo) / (hi - lo), 0, 1)
Nếu lo > hi thì hàm tự đảo chiều (giá trị càng nhỏ càng "giống" lửa/khói).
Đặc trưng thiếu (None / NaN) bị bỏ qua và trọng số được chuẩn hóa lại.

-> Sau này có thể thay toàn bộ module này bằng SVM / Random Forest / CNN.
"""
from __future__ import annotations

import math


def ramp(x: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 1.0 if x >= hi else 0.0
    return min(1.0, max(0.0, (x - lo) / (hi - lo)))


def score_features(features: dict[str, float | None], spec: dict[str, list[float]]) -> float:
    total_w, acc = 0.0, 0.0
    for name, (weight, lo, hi) in spec.items():
        value = features.get(name)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        acc += weight * ramp(float(value), lo, hi)
        total_w += weight
    return acc / total_w if total_w > 0 else 0.0
