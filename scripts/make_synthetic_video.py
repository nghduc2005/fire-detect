"""Sinh video TỔNG HỢP để chạy thử pipeline khi chưa có dữ liệu thật.

Cảnh gồm:
- Nền tĩnh có texture (trời, cỏ, tòa nhà xám)
- Biển báo đỏ ĐỨNG YÊN       -> phải bị loại bởi tầng chuyển động
- Đèn đỏ di chuyển CỨNG       -> phải bị loại bởi đặc trưng thời gian (không flicker)
- Xe xám di chuyển CỨNG       -> phải bị loại bởi đặc trưng khói
- Ngọn lửa nhấp nháy          (video dương tính)
- Cột khói bán trong suốt bốc lên (video dương tính)

CHÚ Ý: chỉ dùng để kiểm tra code chạy đúng, KHÔNG dùng để báo cáo kết quả.

    python scripts/make_synthetic_video.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import src  # noqa: E402,F401  (cấu hình console UTF-8)
W, H, FPS, N_FRAMES = 640, 360, 25, 250
FIRE_START, SMOKE_START = 50, 75


def make_background(rng: np.random.Generator) -> np.ndarray:
    bg = np.zeros((H, W, 3), np.float32)
    horizon = int(H * 0.5)
    for y in range(horizon):  # Trời: gradient xanh
        t = y / horizon
        bg[y] = (210 - 30 * t, 170 - 20 * t, 120 - 20 * t)
    bg[horizon:] = (50, 120, 60)  # Cỏ xanh
    noise = cv2.GaussianBlur(rng.normal(0, 25, (H, W)).astype(np.float32), (0, 0), 2)
    bg += noise[..., None]
    # Cây (texture tối) để khói có "cạnh" mà làm mờ
    for _ in range(25):
        cx, cy = int(rng.integers(0, W)), int(rng.integers(horizon - 40, horizon + 10))
        cv2.circle(bg, (cx, cy), int(rng.integers(10, 30)), (30, 80, 40), -1)
    # Tòa nhà xám (tĩnh) có cửa sổ
    cv2.rectangle(bg, (470, 70), (600, horizon + 20), (150, 150, 150), -1)
    for wy in range(85, horizon, 22):
        for wx in range(482, 590, 26):
            cv2.rectangle(bg, (wx, wy), (wx + 14, wy + 12), (90, 90, 95), -1)
    # Biển báo đỏ đứng yên
    cv2.rectangle(bg, (40, 230), (90, 270), (20, 40, 210), -1)
    return np.clip(bg, 0, 255).astype(np.uint8)


def draw_fire(frame: np.ndarray, t: int, rng: np.random.Generator) -> None:
    base_x, base_y = 330, 300
    grow = min(1.0, 0.4 + (t - FIRE_START) / 100)
    layer = frame.copy()
    for _ in range(7):
        cx = base_x + int(rng.integers(-18, 18) * grow)
        cy = base_y - int(rng.integers(0, 45) * grow)
        axes = (int(rng.integers(8, 22) * grow) + 1, int(rng.integers(20, 50) * grow) + 1)
        color = (int(rng.integers(0, 40)), int(rng.integers(90, 200)), 255)
        cv2.ellipse(layer, (cx, cy), axes, float(rng.integers(-20, 20)), 0, 360, color, -1)
    core = (int(rng.integers(0, 60)), int(rng.integers(200, 240)), 255)
    cv2.ellipse(layer, (base_x, base_y - 10), (int(10 * grow) + 1, int(22 * grow) + 1), 0, 0, 360,
                core, -1)
    frame[:] = cv2.GaussianBlur(layer, (3, 3), 0)


class SmokePlume:
    def __init__(self, rng: np.random.Generator, source=(170, 290)):
        self.rng, self.source, self.puffs = rng, source, []

    def step(self) -> None:
        for _ in range(2):
            self.puffs.append([self.source[0] + self.rng.normal(0, 6), float(self.source[1]),
                               float(self.rng.uniform(8, 14)), 0])
        for p in self.puffs:
            p[0] += self.rng.normal(0.3, 0.8)
            p[1] -= self.rng.uniform(1.2, 2.4)
            p[2] += 0.35
            p[3] += 1
        self.puffs = [p for p in self.puffs if p[3] < 110 and p[1] > -30]

    def render(self, frame: np.ndarray) -> None:
        alpha = np.zeros((H, W), np.float32)
        for x, y, r, age in self.puffs:
            cv2.circle(alpha, (int(x), int(y)), int(r), 0.25 * (1 - age / 110), -1)
        alpha = np.clip(cv2.GaussianBlur(alpha, (0, 0), 6), 0, 0.85)[..., None]
        frame[:] = (frame * (1 - alpha) + 195 * alpha).astype(np.uint8)


def make_video(path: Path, with_fire: bool, with_smoke: bool, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    bg = make_background(rng)
    plume = SmokePlume(rng)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for t in range(N_FRAMES):
        frame = bg.copy()
        # Xe xám di chuyển cứng
        car_x = int((t * 3) % (W + 120)) - 120
        cv2.rectangle(frame, (car_x, 255), (car_x + 90, 290), (140, 140, 140), -1)
        cv2.rectangle(frame, (car_x + 15, 240), (car_x + 70, 258), (120, 120, 125), -1)
        # Đèn đỏ di chuyển cứng (ngược chiều)
        lx = W - int((t * 4) % (W + 60))
        cv2.circle(frame, (lx, 325), 16, (30, 60, 230), -1)

        if with_smoke and t >= SMOKE_START:
            plume.step()
            plume.render(frame)
        if with_fire and t >= FIRE_START:
            draw_fire(frame, t, rng)

        noise = rng.normal(0, 2, frame.shape)
        writer.write(np.clip(frame + noise, 0, 255).astype(np.uint8))
    writer.release()


def main() -> None:
    samples = ROOT / "data" / "samples"
    ann_dir = ROOT / "data" / "annotations"
    samples.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    videos = {
        "synthetic_fire_smoke": (True, True, {"fire": [[FIRE_START, N_FRAMES - 1]],
                                              "smoke": [[SMOKE_START, N_FRAMES - 1]]}),
        "synthetic_negative": (False, False, {"fire": [], "smoke": []}),
    }
    for name, (fire, smoke, ann) in videos.items():
        make_video(samples / f"{name}.mp4", fire, smoke)
        (ann_dir / f"{name}.json").write_text(json.dumps(ann, indent=2), encoding="utf-8")
        print(f"Đã tạo {samples / f'{name}.mp4'} + nhãn")


if __name__ == "__main__":
    main()
