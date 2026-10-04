"""Fire & smoke detection from video - classical image processing pipeline."""
import sys

__version__ = "0.1.0"

# Console Windows mặc định cp1252 -> lỗi khi in tiếng Việt
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure") and (_stream.encoding or "").lower() != "utf-8":
        _stream.reconfigure(encoding="utf-8", errors="replace")
