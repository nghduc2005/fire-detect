"""Demo Streamlit: chạy pipeline trên video, xem mask trung gian, chỉnh ngưỡng trực tiếp.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import load_config  # noqa: E402
from src.detectors import ClassicalDetector  # noqa: E402
from src.video_io import VideoReader  # noqa: E402
from src.visualization import draw_detections, make_debug_grid  # noqa: E402

st.set_page_config(page_title="Phát hiện khói / lửa", page_icon="🔥", layout="wide")
st.title("🔥 Phát hiện khói / lửa từ video")
st.caption("Pipeline xử lý ảnh truyền thống: màu · chuyển động · biến đổi hình dạng theo thời gian · lọc sai")

base = load_config()

# ------------------------------------------------------------------ Sidebar
with st.sidebar:
    st.header("Nguồn video")
    samples = sorted((ROOT / "data" / "samples").glob("*.mp4"))
    source = st.radio("Chọn", ["Video mẫu", "Tải lên"], horizontal=True)
    video_path: str | None = None
    if source == "Video mẫu":
        if samples:
            video_path = str(st.selectbox("Video", samples, format_func=lambda p: p.name))
        else:
            st.info("Chưa có video mẫu. Chạy `python scripts/make_synthetic_video.py`.")
    else:
        up = st.file_uploader("Video", type=["mp4", "avi", "mov", "mkv"])
        if up:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(up.name).suffix)
            tmp.write(up.read())
            video_path = tmp.name

    st.header("Các tầng pipeline")
    pipe = {
        "detect_fire": st.checkbox("Phát hiện lửa", True),
        "detect_smoke": st.checkbox("Phát hiện khói", True),
        "use_motion": st.checkbox("Chuyển động", True),
        "use_temporal": st.checkbox("Đặc trưng thời gian", True),
        "use_persistence": st.checkbox("Bỏ phiếu thời gian", True),
    }

    st.header("Ngưỡng")
    fire_method = st.selectbox("Luật màu lửa", ["combined", "rgb", "ycbcr", "any"])
    r_t = st.slider("R_T (RGB)", 80, 220, base["fire_color"]["rgb"]["r_threshold"])
    tau = st.slider("τ |Cb−Cr| (YCbCr)", 0, 120, base["fire_color"]["ycbcr"]["tau"])
    smoke_diff = st.slider("Chênh lệch kênh tối đa (khói)", 5, 60,
                           base["smoke_color"]["max_channel_diff"])
    min_area = st.slider("Diện tích tối thiểu (px)", 20, 2000, base["candidates"]["min_area"])
    fire_thr = st.slider("Ngưỡng điểm lửa", 0.0, 1.0, base["scoring"]["fire"]["threshold"])
    smoke_thr = st.slider("Ngưỡng điểm khói", 0.0, 1.0, base["scoring"]["smoke"]["threshold"])

    show_debug = st.checkbox("Hiện mask trung gian", True)
    every = st.slider("Hiển thị mỗi N frame", 1, 10, 2)
    max_frames = st.number_input("Số frame tối đa (0 = hết)", 0, 100000, 0)

cfg = load_config(overrides={
    "pipeline": pipe,
    "fire_color": {"method": fire_method, "rgb": {"r_threshold": r_t}, "ycbcr": {"tau": tau}},
    "smoke_color": {"max_channel_diff": smoke_diff},
    "candidates": {"min_area": min_area},
    "scoring": {"fire": {"threshold": fire_thr}, "smoke": {"threshold": smoke_thr}},
})

# ------------------------------------------------------------------ Main
if video_path and st.button("▶ Chạy", type="primary"):
    view = st.empty()
    progress = st.progress(0.0)
    c1, c2, c3 = st.columns(3)
    m_frame, m_fire, m_smoke = c1.empty(), c2.empty(), c3.empty()
    timeline = []

    with VideoReader(video_path, max_frames=max_frames or None) as reader:
        total = max_frames or reader.frame_count or 1
        detector = ClassicalDetector(cfg, fps=reader.fps)
        for idx, frame in reader:
            dets = detector.process(frame)
            fire_s = max((d.score for d in dets if d.label == "fire"), default=0.0)
            smoke_s = max((d.score for d in dets if d.label == "smoke"), default=0.0)
            timeline.append({"frame": idx, "fire": fire_s, "smoke": smoke_s})

            if idx % every == 0:
                vis = draw_detections(frame, dets)
                if show_debug:
                    vis = make_debug_grid(vis, detector.debug, cfg["visualization"]["tile_width"])
                view.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True)
                m_frame.metric("Frame", idx)
                m_fire.metric("Điểm lửa", f"{fire_s:.2f}")
                m_smoke.metric("Điểm khói", f"{smoke_s:.2f}")
            progress.progress(min(1.0, (idx + 1) / total))

    df = pd.DataFrame(timeline).set_index("frame")
    st.subheader("Điểm phát hiện theo thời gian")
    st.line_chart(df)
    st.write(f"Frame có lửa: **{(df.fire > 0).sum()}** · Frame có khói: **{(df.smoke > 0).sum()}** "
             f"/ {len(df)} frame")
