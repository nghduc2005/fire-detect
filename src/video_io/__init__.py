"""Trình đọc / ghi video dùng cho toàn pipeline."""

from .video_reader import VideoReader
from .video_writer import VideoWriter

__all__ = ["VideoReader", "VideoWriter"]
