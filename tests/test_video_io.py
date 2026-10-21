"""Test cho ``src.video_io``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.video_io import VideoReader, VideoWriter


def test_reader_opens_and_reports_metadata(tmp_video_path: Path):
    with VideoReader(tmp_video_path) as reader:
        assert reader.is_opened
        assert reader.fps > 0
        assert reader.width > 0 and reader.height > 0
        assert reader.frame_count == 10
        assert reader.size == (reader.width, reader.height)


def test_reader_iterates_all_frames(tmp_video_path: Path):
    frames = list(VideoReader(tmp_video_path))
    assert len(frames) == 10
    for f in frames:
        assert isinstance(f, np.ndarray)
        assert f.ndim == 3 and f.shape[2] == 3


def test_reader_read_method(tmp_video_path: Path):
    reader = VideoReader(tmp_video_path)
    try:
        ok, frame = reader.read()
        assert ok and frame is not None
    finally:
        reader.release()


def test_reader_raises_on_missing_file(tmp_path: Path):
    with pytest.raises(IOError):
        VideoReader(tmp_path / "nope.mp4")


def test_writer_lazy_init_and_count(tmp_path: Path, tiny_frame: np.ndarray):
    out = tmp_path / "written.mp4"
    with VideoWriter(out, fps=20.0) as writer:
        for _ in range(5):
            writer.write(tiny_frame)
        assert writer.frames_written == 5
        assert writer.size == (tiny_frame.shape[1], tiny_frame.shape[0])
    assert out.exists() and out.stat().st_size > 0


def test_writer_resizes_mismatched_frame(tmp_path: Path, tiny_frame: np.ndarray):
    out = tmp_path / "mixed.mp4"
    writer = VideoWriter(out, fps=20.0)
    writer.write(tiny_frame)  # size = (80, 60)
    # Frame khác size — writer phải tự resize về size đầu tiên, không crash.
    big = np.zeros((120, 160, 3), dtype=np.uint8)
    writer.write(big)
    writer.release()
    assert writer.frames_written == 2
    assert writer.size == (80, 60)


def test_round_trip_reader_writer(tmp_path: Path, tiny_frame: np.ndarray):
    out = tmp_path / "rt.mp4"
    with VideoWriter(out, fps=10.0) as writer:
        for _ in range(6):
            writer.write(tiny_frame)
    with VideoReader(out) as reader:
        assert reader.frame_count == 6
        count = sum(1 for _ in reader)
    assert count == 6
