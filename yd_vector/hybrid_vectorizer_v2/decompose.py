from __future__ import annotations

from yd_vector.hybrid_vectorizer.cleanup import douglas_peucker_closed, merge_near_duplicate_points
from yd_vector.hybrid_vectorizer.geometry import ClosedContour, ContourRegion, Point
from yd_vector.hybrid_vectorizer_v2.config import HybridVectorizerV2Config
from yd_vector.hybrid_vectorizer_v2.ellipse_subshape import (
    attach_ellipse_subshape_anchors,
    detect_bottom_ellipse_subshape,
    required_anchor_indices,
)
from yd_vector.hybrid_vectorizer_v2.geometry import ContourPartPlan, RegionDecomposition
from yd_vector.hybrid_vectorizer_v2.primitive_detection import detect_primitive_candidates, detect_straight_runs


def decompose_region(region: ContourRegion, config: HybridVectorizerV2Config) -> RegionDecomposition:
    return RegionDecomposition(
        region=region,
        outer=_decompose_contour(region.outer, role="outer", polarity="positive", config=config),
        holes=[
            _decompose_contour(hole, role="hole", polarity="negative", config=config)
            for hole in region.holes
        ],
    )


def _decompose_contour(
    contour: ClosedContour,
    role: str,
    polarity: str,
    config: HybridVectorizerV2Config,
) -> ContourPartPlan:
    primitive_candidates = detect_primitive_candidates(contour, polarity=polarity, role=role, config=config)
    ellipse_subshape = detect_bottom_ellipse_subshape(contour, config) if role == "outer" else None
    anchor_indices = _build_anchor_indices(contour.points, config, required_anchor_indices(ellipse_subshape))
    anchor_points = [contour.points[index] for index in anchor_indices]
    ellipse_subshape = attach_ellipse_subshape_anchors(ellipse_subshape, anchor_indices)
    straight_runs = detect_straight_runs(contour.points, anchor_points, anchor_indices, config)
    return ContourPartPlan(
        contour=contour,
        role=role,
        polarity=polarity,
        anchor_points=anchor_points,
        anchor_indices=anchor_indices,
        straight_runs=straight_runs,
        primitive_candidates=primitive_candidates,
        ellipse_subshape=ellipse_subshape,
    )


def _build_anchor_indices(
    source_points: list[Point],
    config: HybridVectorizerV2Config,
    required_indices: list[int] | None = None,
) -> list[int]:
    anchor_points = merge_near_duplicate_points(source_points, max(0.01, config.merge_distance * 0.5))
    anchor_points = douglas_peucker_closed(anchor_points, tolerance=max(0.25, config.simplify_tolerance))
    if len(anchor_points) < 4:
        anchor_points = source_points[:]
    anchor_indices = _match_anchor_indices(source_points, anchor_points)
    if required_indices:
        anchor_indices = _merge_required_indices(anchor_indices, required_indices, len(source_points))
    return anchor_indices


def _match_anchor_indices(source_points: list[Point], anchor_points: list[Point]) -> list[int]:
    if not anchor_points:
        return []
    indices: list[int] = []
    search_start = 0
    source_count = len(source_points)
    for anchor in anchor_points:
        found_index = None
        for step in range(source_count):
            index = (search_start + step) % source_count
            point = source_points[index]
            if abs(point.x - anchor.x) <= 1e-6 and abs(point.y - anchor.y) <= 1e-6:
                found_index = index
                search_start = (index + 1) % source_count
                break
        if found_index is None:
            found_index = min(
                range(source_count),
                key=lambda idx: (source_points[idx].x - anchor.x) ** 2 + (source_points[idx].y - anchor.y) ** 2,
            )
            search_start = (found_index + 1) % source_count
        indices.append(found_index)
    return indices


def _merge_required_indices(anchor_indices: list[int], required_indices: list[int], count: int) -> list[int]:
    merged = set(anchor_indices)
    merged.update(index % count for index in required_indices)
    return [index for index in range(count) if index in merged]
