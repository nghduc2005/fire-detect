"""scripts/make_synthetic_dataset.py
Sinh bộ video demo nhỏ để chạy thử pipeline / Ablation Study khi chưa có
dataset thật (Bilkent, MIVIA, FIRESENSE).

    python -m scripts.make_synthetic_dataset --output data/raw

Video sinh ra (MJPG .avi, 90 frame, 30 FPS, 640x480):
    pos_smoke.avi        - cột khói xám bốc lên, loang rộng dần
    pos_fire.avi         - ngọn lửa cam nhấp nháy, diện tích thay đổi
    neg_traffic.avi      - đèn giao thông tĩnh + ô tô trắng chạy qua
    neg_sunset.avi       - hoàng hôn cam + mảng mây xám tĩnh
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

W, H, FPS, N_FRAMES = 640, 480, 30, 90


def _background(rng: np.random.Generator, base: tuple[int, int, int]) -> np.ndarray:
    """Tạo nền gradient + nhiễu hạt nhẹ (BGR)."""
    yy = np.linspace(0.75, 1.05, H, dtype=np.float32)[:, None, None]
    img = np.clip(np.array(base, dtype=np.float32)[None, None, :] * yy, 0, 255)
    img = np.repeat(img, W, axis=1)
    noise = rng.normal(0, 3.0, (H, W, 1))
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def _write(path: str, frames) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(path, fourcc, FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Không mở được VideoWriter cho {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()
    print(f"  -> {path}")


def make_smoke(rng: np.random.Generator):
    bg = _background(rng, (48, 52, 58))
    for t in range(N_FRAMES):
        frame = bg.copy()
        if t >= 6:  # 6 frame đầu không có khói để khởi tạo nền sạch
            layer = np.zeros((H, W), dtype=np.float32)
            cx = int(W / 2 + 25 * np.sin(t / 9.0))
            cy = int(H - 90 - 2.4 * (t - 6))
            radius = int(42 + 0.85 * (t - 6))
            for k in range(5):
                ox = int(rng.normal(0, 12))
                oy = int(rng.normal(0, 10))
                cv2.ellipse(
                    layer,
                    (cx + ox, cy + oy + k * radius // 3),
                    (radius, int(radius * 0.75)),
                    0,
                    0,
                    360,
                    1.0,
                    -1,
                )
            layer = cv2.GaussianBlur(layer, (0, 0), 16)
            layer = np.clip(layer, 0, 1)[..., None]
            smoke_color = np.array([176, 178, 180], dtype=np.float32)
            frame = (frame * (1 - 0.55 * layer) + smoke_color * 0.55 * layer).astype(np.uint8)
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        yield frame


def make_fire(rng: np.random.Generator):
    bg = _background(rng, (30, 28, 34))
    base_x, base_y = W // 2, H - 80
    for t in range(N_FRAMES):
        frame = bg.copy()
        if t >= 6:
            glow = np.zeros((H, W), dtype=np.uint8)
            flick = 1.0 + 0.18 * np.sin(t * 1.7) + float(rng.normal(0, 0.05))
            n_lobes = 6
            for i in range(n_lobes):
                frac = i / max(1, n_lobes - 1)
                y = int(base_y - frac * (110 + 25 * flick))
                x = int(base_x + (i - n_lobes / 2) * 7 + rng.normal(0, 5))
                radius_x = max(6, int((34 - 20 * frac) * flick + rng.normal(0, 2)))
                radius_y = max(10, int((52 - 25 * frac) * flick + rng.normal(0, 3)))
                color = (
                    int(20 + 70 * frac),
                    int(90 + 120 * frac),
                    int(230 - 30 * frac),
                )
                cv2.ellipse(frame, (x, y), (radius_x, radius_y), 0, 0, 360, color, -1)
                cv2.ellipse(glow, (x, y), (radius_x * 2, radius_y * 2), 0, 0, 360, (0, 90, 180), -1)
            glow = cv2.GaussianBlur(glow, (0, 0), 21)
            glow_bgr = cv2.cvtColor(glow, cv2.COLOR_GRAY2BGR)
            frame = cv2.addWeighted(frame, 1.0, glow_bgr, 0.35, 0)
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        yield frame


def make_traffic(rng: np.random.Generator):
    bg = _background(rng, (30, 32, 34))
    for t in range(N_FRAMES):
        frame = bg.copy()
        # Hai đèn giao thông tĩnh
        cv2.circle(frame, (150, 150), 26, (0, 210, 240), -1)
        cv2.circle(frame, (500, 120), 22, (0, 0, 230), -1)
        # Đèn đường nhấp nháy
        lamp_val = 235 if (t // 7) % 2 == 0 else 160
        cv2.circle(frame, (320, 70), 18, (lamp_val, lamp_val, lamp_val), -1)
        # Ô tô trắng chạy ngang (đèn pha/ thân xe sáng, ít bão hoà)
        car_x = int((t * 11) % (W + 120)) - 120
        cv2.rectangle(frame, (car_x, 360), (car_x + 110, 405), (225, 228, 232), -1)
        cv2.circle(frame, (car_x + 100, 385), 10, (255, 255, 255), -1)
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        yield frame


def make_sunset(rng: np.random.Generator):
    for t in range(N_FRAMES):
        frame = np.zeros((H, W, 3), dtype=np.float32)
        for y in range(H):
            frac = y / H
            top = np.array([40, 90, 210], dtype=np.float32)
            bottom = np.array([120, 150, 190], dtype=np.float32)
            frame[y, :, :] = top * (1 - frac) + bottom * frac
        # Mảng mây xám tĩnh
        cloud = np.zeros((H, W), dtype=np.float32)
        for k, (cx, cy, rx, ry) in enumerate(
            [(160, 170, 120, 38), (430, 130, 150, 30), (300, 240, 170, 34)]
        ):
            cv2.ellipse(cloud, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
        cloud = cv2.GaussianBlur(cloud, (0, 0), 18)[..., None]
        gray = np.array([165, 168, 172], dtype=np.float32)
        frame = frame * (1 - 0.55 * cloud) + gray * 0.55 * cloud
        noise = rng.normal(0, 2.0, (H, W, 1))
        frame = np.clip(frame + noise, 0, 255).astype(np.uint8)
        yield frame


GENERATORS = {
    "pos_smoke.avi": make_smoke,
    "pos_fire.avi": make_fire,
    "neg_traffic.avi": make_traffic,
    "neg_sunset.avi": make_sunset,
}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Sinh video demo tổng hợp")
    parser.add_argument("--output", default=os.path.join("data", "raw"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    os.makedirs(args.output, exist_ok=True)
    print(f"Sinh video demo vào {args.output}:")
    for name, factory in GENERATORS.items():
        rng = np.random.default_rng(args.seed)
        _write(os.path.join(args.output, name), factory(rng))


if __name__ == "__main__":
    main()
