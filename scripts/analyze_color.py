"""Phân tích phân bố màu của pixel lửa/khói thật để BIỆN MINH cho ngưỡng màu.

Đầu vào: thư mục ảnh + thư mục mask nhị phân cùng tên (pixel trắng = lửa/khói).
Đầu ra : histogram 2D Cb-Cr và H-S của pixel dương tính so với âm tính.

Ví dụ:
    python scripts/analyze_color.py --images data/color/images --masks data/color/masks
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src  # noqa: E402,F401  (cấu hình console UTF-8)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_pixels(images: Path, masks: Path, max_per_image: int = 20000):
    rng = np.random.default_rng(0)
    pos_ycc, neg_ycc, pos_hsv, neg_hsv = [], [], [], []
    for ip in sorted(p for p in images.iterdir() if p.suffix.lower() in IMG_EXTS):
        mp = next((masks / f"{ip.stem}{e}" for e in IMG_EXTS if (masks / f"{ip.stem}{e}").exists()),
                  None)
        if mp is None:
            continue
        img = cv2.imread(str(ip))
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if img is None or m is None:
            continue
        m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 127
        ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).reshape(-1, 3)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        flat = m.reshape(-1)
        for sel, ycc_list, hsv_list in ((flat, pos_ycc, pos_hsv), (~flat, neg_ycc, neg_hsv)):
            idx = np.flatnonzero(sel)
            if idx.size > max_per_image:
                idx = rng.choice(idx, max_per_image, replace=False)
            ycc_list.append(ycc[idx])
            hsv_list.append(hsv[idx])
    cat = lambda xs: np.concatenate(xs) if xs else np.empty((0, 3), np.uint8)  # noqa: E731
    return cat(pos_ycc), cat(neg_ycc), cat(pos_hsv), cat(neg_hsv)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--masks", required=True)
    p.add_argument("--out", default="outputs/color_analysis.png")
    args = p.parse_args()

    pos_ycc, neg_ycc, pos_hsv, neg_hsv = collect_pixels(Path(args.images), Path(args.masks))
    if pos_ycc.size == 0:
        raise SystemExit("Không tìm thấy cặp ảnh/mask hợp lệ.")

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    panels = [
        (pos_ycc[:, 2], pos_ycc[:, 1], "Dương tính: Cb-Cr", "Cb", "Cr", (0, 255), (0, 255)),
        (neg_ycc[:, 2], neg_ycc[:, 1], "Âm tính: Cb-Cr", "Cb", "Cr", (0, 255), (0, 255)),
        (pos_hsv[:, 0], pos_hsv[:, 1], "Dương tính: H-S", "H", "S", (0, 180), (0, 255)),
        (neg_hsv[:, 0], neg_hsv[:, 1], "Âm tính: H-S", "H", "S", (0, 180), (0, 255)),
    ]
    for ax, (x, y, title, xl, yl, xr, yr) in zip(axes.flat, panels):
        ax.hist2d(x, y, bins=64, range=[xr, yr], cmap="inferno", cmin=1)
        ax.set(title=title, xlabel=xl, ylabel=yl)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Đã lưu: {args.out}  ({len(pos_ycc)} pixel dương, {len(neg_ycc)} pixel âm)")


if __name__ == "__main__":
    main()
