"""Unit test cho module hậu xử lý của Thành viên 4."""

from __future__ import annotations

import numpy as np

from src.detectors.base import Region
from src.filtering import (
    FeatureScorer,
    PersistenceVoter,
    extract_candidates,
    morphology_clean,
)


def test_morphology_clean_removes_single_pixel_noise() -> None:
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10, 10] = 255
    mask[25:40, 20:35] = 255
    cleaned = morphology_clean(mask, open_ksize=3, close_ksize=5)
    assert cleaned[10, 10] == 0
    assert cleaned[30, 25] == 255


def test_extract_candidates_geometry_filter_and_extras() -> None:
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[20:80, 20:80] = 255       # vùng hợp lệ
    mask[120:125, 120:125] = 255   # quá nhỏ
    regions = extract_candidates(mask, "fire", {"min_area": 200, "max_aspect": 4.0})
    assert len(regions) == 1
    region = regions[0]
    assert region.area == 3600
    assert region.mask is not None and region.mask.shape == (60, 60)
    assert region.extras["solidity"] > 0.9
    assert abs(region.extras["area_ratio"] - 3600 / 40000) < 1e-6
    assert "boundary_roughness" in region.extras


def test_extract_candidates_rejects_far_from_square() -> None:
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[10:40, 10:190] = 255   # aspect ~6 > 4
    assert extract_candidates(mask, "smoke", {"min_area": 100, "max_aspect": 4.0}) == []


def test_feature_scorer_linear_and_reversed_range() -> None:
    scorer = FeatureScorer(
        {
            "threshold": 0.4,
            "features": {
                "edge_blur": [0.5, 1.0, 0.5],   # thang điểm đảo
                "solidity": [0.5, 0.2, 1.0],
            },
        },
        kind="smoke",
    )
    region = Region(bbox=(0, 0, 10, 10), area=100, centroid=(5.0, 5.0), label="smoke")
    region.extras.update({"edge_blur": 0.5, "solidity": 0.6})
    score, breakdown = scorer.score(region)
    assert breakdown["edge_blur"] == 1.0
    assert abs(breakdown["solidity"] - 0.5) < 1e-6
    assert abs(score - 0.75) < 1e-6
    assert scorer.threshold == 0.4


def test_feature_scorer_missing_feature_is_neutral() -> None:
    scorer = FeatureScorer({"features": {"flicker": [1.0, 0.0, 1.0]}})
    score, breakdown = scorer.score(
        Region(bbox=(0, 0, 5, 5), area=25, centroid=(2.5, 2.5), label="fire")
    )
    assert breakdown["flicker"] == 0.5
    assert score == 0.5


def test_persistence_voter_needs_min_hits_and_records_misses() -> None:
    voter = PersistenceVoter({"window": 5, "min_hits": 3})
    region = Region(bbox=(0, 0, 10, 10), area=100, centroid=(5, 5), label="smoke", track_id=1)

    outcomes = []
    for _ in range(3):
        voter.begin_frame()
        outcomes.append(voter.vote(region))
        voter.end_frame({1})
    assert outcomes == [False, False, True]

    # Track vắng mặt 3 frame -> cửa sổ trượt ghi 0, không còn đủ min_hits
    for _ in range(3):
        voter.begin_frame()
        voter.end_frame({1})
    voter.begin_frame()
    assert voter.vote(region) is False


def test_persistence_voter_without_tracker_and_reset() -> None:
    voter = PersistenceVoter({"window": 3, "min_hits": 2})
    region = Region(bbox=(0, 0, 5, 5), area=25, centroid=(2, 2), label="fire", track_id=-1)
    assert voter.vote(region) is False

    voter.reset()
    region.track_id = 7
    voter.begin_frame()
    voter.vote(region)
    voter.end_frame({7})
    assert voter.vote(region) is True
