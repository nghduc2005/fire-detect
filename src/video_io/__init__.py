"""Đọc / ghi video. (Đặt tên `video_io` để tránh trùng module chuẩn `io`.)"""
from .video_reader import VideoReader
from .video_writer import VideoWriter

__all__ = ["VideoReader", "VideoWriter"]
