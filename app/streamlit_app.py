"""
Giao diện demo trực quan cho hệ thống phát hiện khói / lửa.

Chạy::

    streamlit run app/streamlit_app.py

Tính năng:

- Upload video hoặc chọn file có sẵn trong ``data/samples/``.
- Thanh trượt điều chỉnh ngưỡng chính (resize, R_T, S_T, S_smoke, min_area, ...).
- Xem từng frame kèm bbox phát hiện.
- Chế độ **Debug**: xem song song các mask trung gian (fire_color, smoke_color,
  motion, fire_clean, smoke_clean) → giúp TV4 tinh chỉnh và minh họa báo cáo.
- Thống kê FPS trung bình.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

# Cho phép ``streamlit run app/streamlit_app.py`` tìm được package src.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from src.detectors import ClassicalDetector  # noqa: E402
from src.video_io import VideoReader  # noqa: E402
from src.visualization import draw_detections, mask_to_bgr  # noqa: E402


# ====================================================================== #
# Trang và tiêu đề
# ====================================================================== #
st.set_page_config(
    page_title="Fire / Smoke Detector",
    page_icon="🔥",
    layout="wide",
)
st.title("🔥 Hệ thống phát hiện Khói / Lửa từ Video")
st.caption("Bài tập lớn môn Xử lý ảnh — PTIT. Pipeline xử lý ảnh truyền thống (XLA).")


# ====================================================================== #
# Sidebar — cấu hình
# ====================================================================== #
def build_sidebar_config():
    """Dựng ConfigDict từ các slider của sidebar (ghi đè lên default.yaml)."""
    cfg = load_config(DEFAULT_CONFIG_PATH)

    with st.sidebar:
        st.header("Cấu hình pipeline")

        st.subheader("Tiền xử lý")
        cfg.preprocessing.resize_width = st.slider(
            "Resize width", 320, 1280, int(cfg.preprocessing.resize_width), step=32
        )
        cfg.preprocessing.blur_ksize = st.slider(
            "Gaussian blur ksize", 0, 15, int(cfg.preprocessing.blur_ksize), step=1
        )

        st.subheader("Màu lửa (Chen 2004 + Çelik 2009)")
        cfg.fire_color.rgb.r_threshold = st.slider(
            "R_T", 50, 220, int(cfg.fire_color.rgb.r_threshold)
        )
        cfg.fire_color.rgb.s_threshold = st.slider(
            "S_T", 10, 150, int(cfg.fire_color.rgb.s_threshold)
        )
        cfg.fire_color.ycbcr.tau = st.slider(
            "|Cb - Cr| tau", 0, 100, int(cfg.fire_color.ycbcr.tau)
        )

        st.subheader("Màu khói")
        cfg.smoke_color.max_channel_diff = st.slider(
            "Max channel diff α", 0, 60, int(cfg.smoke_color.max_channel_diff)
        )
        cfg.smoke_color.s_max = st.slider(
            "S_smoke max", 0, 150, int(cfg.smoke_color.s_max)
        )

        st.subheader("Ứng viên")
        cfg.candidates.min_area = st.slider(
            "Min area (pixel)", 10, 2000, int(cfg.candidates.min_area), step=10
        )

        st.subheader("Bật / tắt tầng (ablation)")
        cfg.pipeline.detect_fire = st.checkbox("Phát hiện lửa", bool(cfg.pipeline.detect_fire))
        cfg.pipeline.detect_smoke = st.checkbox("Phát hiện khói", bool(cfg.pipeline.detect_smoke))
        cfg.pipeline.use_motion = st.checkbox("Chuyển động (AND)", bool(cfg.pipeline.use_motion))
        cfg.pipeline.use_temporal = st.checkbox("Đặc trưng thời gian", bool(cfg.pipeline.use_temporal))
        cfg.pipeline.use_persistence = st.checkbox(
            "Persistence voting", bool(cfg.pipeline.use_persistence)
        )

        st.divider()
        debug = st.checkbox("Hiện mask trung gian (debug)", value=True)
        max_frames = st.slider("Giới hạn số frame", 10, 2000, 300, step=10)
    return cfg, debug, max_frames


# ====================================================================== #
# Nguồn video — upload hoặc samples
# ====================================================================== #
def select_source() -> Path | None:
    st.subheader("1. Chọn nguồn video")

    samples_dir = _ROOT / "data" / "samples"
    sample_files = sorted(
        [p for p in samples_dir.glob("*") if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}]
    ) if samples_dir.exists() else []

    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader(
            "Upload file video", type=["mp4", "avi", "mov", "mkv"]
        )
    with col2:
        selected = None
        if sample_files:
            names = ["— chọn —"] + [p.name for p in sample_files]
            idx = st.selectbox("Hoặc chọn mẫu có sẵn", options=range(len(names)), format_func=lambda i: names[i])
            if idx > 0:
                selected = sample_files[idx - 1]
        else:
            st.info("Chưa có file mẫu trong `data/samples/`.")

    if uploaded is not None:
        suffix = Path(uploaded.name).suffix
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded.read())
        tmp.flush()
        return Path(tmp.name)
    return selected


# ====================================================================== #
# Chạy pipeline
# ====================================================================== #
def run_pipeline(video_path: Path, cfg, debug: bool, max_frames: int) -> None:
    detector = ClassicalDetector(cfg)

    with VideoReader(video_path) as reader:
        st.caption(
            f"**{video_path.name}** · {reader.width}×{reader.height} · "
            f"{reader.fps:.1f} fps · {reader.frame_count} frames"
        )

        main_col, mask_col = st.columns([2, 3]) if debug else (st.container(), None)
        with main_col:
            frame_slot = st.empty()
            info_slot = st.empty()

        mask_slots = {}
        if debug and mask_col is not None:
            with mask_col:
                st.markdown("**Mask trung gian**")
                top = st.columns(2)
                bot = st.columns(3)
                mask_slots["fire_color"] = top[0].empty()
                mask_slots["smoke_color"] = top[1].empty()
                mask_slots["motion"] = bot[0].empty()
                mask_slots["fire_clean"] = bot[1].empty()
                mask_slots["smoke_clean"] = bot[2].empty()

        t_start = time.perf_counter()
        n = 0
        n_det = 0
        for frame in reader:
            t0 = time.perf_counter()
            detections = detector.process(frame)
            dt = time.perf_counter() - t0

            bundle = detector.preprocess_only(frame)
            vis = draw_detections(bundle.bgr, detections)
            frame_slot.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), channels="RGB")

            info_slot.markdown(
                f"Frame {n + 1} · xử lý {dt * 1000:.1f} ms "
                f"(~{1 / max(dt, 1e-6):.1f} fps) · "
                f"{len(detections)} detection(s)"
            )

            if debug and mask_slots:
                for name, slot in mask_slots.items():
                    m = detector.debug_masks.get(name)
                    if m is None:
                        continue
                    slot.image(mask_to_bgr(m), caption=name, channels="BGR", width="stretch")

            n += 1
            n_det += len(detections)
            if n >= max_frames:
                break

        total = time.perf_counter() - t_start
        st.success(
            f"Hoàn thành: **{n}** frames, **{n_det}** detections, "
            f"tổng **{total:.2f}s**, trung bình **{n / max(total, 1e-6):.1f} fps**."
        )


# ====================================================================== #
# Entry
# ====================================================================== #
def main() -> None:
    cfg, debug, max_frames = build_sidebar_config()
    video_path = select_source()

    st.subheader("2. Chạy pipeline")
    run_btn = st.button("▶️ Chạy", type="primary", disabled=video_path is None)
    if run_btn and video_path is not None:
        run_pipeline(video_path, cfg, debug, max_frames)
    elif video_path is None:
        st.info("Hãy upload hoặc chọn một video rồi bấm **Chạy**.")


if __name__ == "__main__":
    main()
