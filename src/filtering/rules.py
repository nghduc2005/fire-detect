"""Các luật lọc phát hiện sai dựa trên hình học và tính bền vững theo thời gian."""
from __future__ import annotations

from src.structures import Region
from src.tracking import Track


def passes_geometry(region: Region, cfg: dict) -> bool:
    """Loại vùng có tỉ lệ khung bất thường (vd. vệt sáng dài, đường kẻ)."""
    _, _, w, h = region.bbox
    if h == 0:
        return False
    aspect = w / h
    return cfg.get("min_aspect", 0.1) <= aspect <= cfg.get("max_aspect", 6.0)


def is_persistent(track: Track, window: int = 15, min_hits: int = 8) -> bool:
    """Vùng phải xuất hiện >= min_hits lần trong `window` frame gần nhất
    (bỏ phiếu trượt) -> loại các nhiễu chỉ xuất hiện thoáng qua."""
    recent = list(track.hits)[-window:]
    return sum(recent) >= min_hits
