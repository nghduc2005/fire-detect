"""Trích vùng ứng viên (connected components) từ mask nhị phân."""
from __future__ import annotations

import cv2
import numpy as np

from src.structures import Region


def extract_regions(mask: np.ndarray, label: str, min_area: float = 150,
                    max_area: float | None = None) -> list[Region]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions: list[Region] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or (max_area is not None and area > max_area):
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        x, y, w, h = cv2.boundingRect(c)
        regions.append(Region(
            label=label,
            bbox=(int(x), int(y), int(w), int(h)),
            area=float(area),
            perimeter=float(cv2.arcLength(c, True)),
            centroid=(m["m10"] / m["m00"], m["m01"] / m["m00"]),
            contour=c,
        ))
    return regions


def region_mask(shape: tuple[int, int], region: Region) -> np.ndarray:
    """Mask nhị phân (uint8) của riêng một vùng, cắt theo bbox."""
    x, y, w, h = region.bbox
    m = np.zeros((h, w), np.uint8)
    cv2.drawContours(m, [region.contour - np.array([[x, y]])], -1, 255, thickness=cv2.FILLED)
    return m
