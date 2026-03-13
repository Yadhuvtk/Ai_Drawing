from __future__ import annotations

from math import hypot

from yd_vector.hybrid_vectorizer.geometry import ClosedContour, Point
from yd_vector.hybrid_vectorizer.loop_builder import (
    build_circle_loop,
    build_ellipse_loop,
    build_rectangle_loop,
    build_rounded_rectangle_loop,
)
from yd_vector.hybrid_vectorizer.shape_analysis import fit_circle, fit_rectangle, fit_rotated_ellipse, fit_rounded_rectangle
from yd_vector.hybrid_vectorizer_v2.config import HybridVectorizerV2Config
from yd_vector.hybrid_vectorizer_v2.geometry import PrimitiveCandidate, StraightRunCandidate


def detect_primitive_candidates(
    contour: ClosedContour,
    polarity: str,
    role: str,
    config: HybridVectorizerV2Config,
) -> list[PrimitiveCandidate]:
    candidates: list[PrimitiveCandidate] = []
    confidence_bias = 0.04 if role == "hole" and config.prefer_hole_primitives else 0.0

    circle = fit_circle(contour.points)
    if circle is not None and _accept_circle(circle, contour, role, config):
        confidence = min(1.0, circle.confidence + confidence_bias)
        loop = build_circle_loop(
            loop_id=contour.contour_id,
            circle=circle.primitive,
            polarity=polarity,
            source_contour_id=contour.contour_id,
            confidence=confidence,
        )
        candidates.append(
            PrimitiveCandidate(
                kind="circle",
                confidence=confidence,
                score=confidence + 0.05,
                loop=loop,
                notes="High-confidence circle fit",
            )
        )

    ellipse = fit_rotated_ellipse(contour.points)
    if ellipse is not None and _accept_ellipse(ellipse, role, config):
        confidence = min(1.0, ellipse.confidence + confidence_bias)
        loop = build_ellipse_loop(
            loop_id=contour.contour_id,
            ellipse=ellipse.primitive,
            polarity=polarity,
            source_contour_id=contour.contour_id,
            confidence=confidence,
        )
        candidates.append(
            PrimitiveCandidate(
                kind="ellipse",
                confidence=confidence,
                score=confidence + 0.03,
                loop=loop,
                notes="High-confidence ellipse fit",
            )
        )

    rounded_rectangle = fit_rounded_rectangle(contour.points)
    if rounded_rectangle is not None and _accept_rounded_rectangle(rounded_rectangle, role, config):
        confidence = rounded_rectangle.confidence
        loop = build_rounded_rectangle_loop(
            loop_id=contour.contour_id,
            rounded_rectangle=rounded_rectangle.primitive,
            polarity=polarity,
            source_contour_id=contour.contour_id,
            confidence=confidence,
        )
        candidates.append(
            PrimitiveCandidate(
                kind="rounded_rectangle",
                confidence=confidence,
                score=confidence + 0.02,
                loop=loop,
                notes="Rounded rectangle candidate",
            )
        )

    rectangle = fit_rectangle(contour.points)
    if rectangle is not None and _accept_rectangle(rectangle, role, config):
        confidence = rectangle.confidence
        loop = build_rectangle_loop(
            loop_id=contour.contour_id,
            rectangle=rectangle.primitive,
            polarity=polarity,
            source_contour_id=contour.contour_id,
            confidence=confidence,
        )
        candidates.append(
            PrimitiveCandidate(
                kind="rectangle",
                confidence=confidence,
                score=confidence,
                loop=loop,
                notes="Rectangle candidate",
            )
        )

    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates


def detect_straight_runs(
    source_points: list[Point],
    anchor_points: list[Point],
    anchor_indices: list[int],
    config: HybridVectorizerV2Config,
) -> list[StraightRunCandidate]:
    if len(anchor_points) < 2 or len(anchor_points) != len(anchor_indices):
        return []

    runs: list[StraightRunCandidate] = []
    count = len(anchor_points)
    for index in range(count):
        next_index = (index + 1) % count
        span = _closed_span(source_points, anchor_indices[index], anchor_indices[next_index])
        if len(span) < 2:
            continue
        mean_error, max_error = _line_errors(span)
        length = hypot(
            anchor_points[next_index].x - anchor_points[index].x,
            anchor_points[next_index].y - anchor_points[index].y,
        )
        if (
            max_error <= config.line_fit_tolerance
            and mean_error <= config.line_fit_tolerance * 0.35
            and length >= max(4.0, config.simplify_tolerance * 3.0)
        ):
            runs.append(
                StraightRunCandidate(
                    start_anchor=index,
                    end_anchor=next_index,
                    mean_error=mean_error,
                    max_error=max_error,
                    length=length,
                )
            )
    return runs


def _accept_circle(result: object, contour: ClosedContour, role: str, config: HybridVectorizerV2Config) -> bool:
    aspect = contour.bbox.width / max(1e-6, contour.bbox.height)
    aspect_min = 0.84 if role == "hole" else 0.90
    aspect_max = 1.16 if role == "hole" else 1.10
    return (
        aspect_min <= aspect <= aspect_max
        and result.radial_error <= config.circle_fit_tolerance
        and result.circularity >= config.circle_circularity_min
        and 0.80 <= result.area_ratio <= 1.20
        and result.confidence >= (config.circle_confidence_threshold - (0.03 if role == "hole" else 0.0))
    )


def _accept_ellipse(result: object, role: str, config: HybridVectorizerV2Config) -> bool:
    confidence_threshold = config.ellipse_confidence_threshold - (0.03 if role == "hole" else 0.0)
    return (
        result.normalized_error <= config.ellipse_fit_tolerance
        and 0.72 <= result.area_ratio <= 1.30
        and result.confidence >= confidence_threshold
    )


def _accept_rectangle(result: object, role: str, config: HybridVectorizerV2Config) -> bool:
    if role == "hole":
        return False
    return (
        result.mean_error <= config.primitive_fit_error_threshold
        and 0.82 <= result.area_ratio <= 1.18
        and result.confidence >= config.rectangle_confidence_threshold
    )


def _accept_rounded_rectangle(result: object, role: str, config: HybridVectorizerV2Config) -> bool:
    if role == "hole":
        return False
    min_dimension = min(result.primitive.width, result.primitive.height)
    radius_ratio = result.primitive.corner_radius / max(1e-6, min_dimension)
    return (
        result.mean_error <= config.primitive_fit_error_threshold
        and 0.74 <= result.area_ratio <= 1.26
        and 0.05 <= radius_ratio <= 0.30
        and result.confidence >= config.rounded_rectangle_confidence_threshold
    )


def _closed_span(points: list[Point], start_index: int, end_index: int) -> list[Point]:
    if start_index <= end_index:
        return points[start_index : end_index + 1]
    return points[start_index:] + points[: end_index + 1]


def _line_errors(points: list[Point]) -> tuple[float, float]:
    if len(points) <= 2:
        return 0.0, 0.0
    start = points[0]
    end = points[-1]
    dx = end.x - start.x
    dy = end.y - start.y
    denom = max(1e-6, dx * dx + dy * dy)
    errors: list[float] = []
    for point in points[1:-1]:
        t = ((point.x - start.x) * dx + (point.y - start.y) * dy) / denom
        t = max(0.0, min(1.0, t))
        proj_x = start.x + dx * t
        proj_y = start.y + dy * t
        errors.append(hypot(point.x - proj_x, point.y - proj_y))
    if not errors:
        return 0.0, 0.0
    return (sum(errors) / len(errors), max(errors))
