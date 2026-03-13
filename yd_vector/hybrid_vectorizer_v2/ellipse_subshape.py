from __future__ import annotations

from dataclasses import replace
from math import atan2, cos, radians, sin

import numpy as np

from yd_vector.hybrid_vectorizer.geometry import ClosedContour, Point, PrimitiveEllipse, SegmentArcElliptical
from yd_vector.hybrid_vectorizer.shape_analysis import fit_rotated_ellipse
from yd_vector.hybrid_vectorizer_v2.config import HybridVectorizerV2Config
from yd_vector.hybrid_vectorizer_v2.geometry import EllipseSubshapeCandidate


def detect_bottom_ellipse_subshape(
    contour: ClosedContour,
    config: HybridVectorizerV2Config,
) -> EllipseSubshapeCandidate | None:
    if contour.area < config.min_ellipse_subshape_area or len(contour.points) < 16:
        return None

    bbox = contour.bbox
    height = max(1e-6, bbox.height)
    width = max(1e-6, bbox.width)
    threshold_y = bbox.min_y + height * config.ellipse_subshape_band_ratio
    runs = _contiguous_runs([point.y >= threshold_y for point in contour.points])
    if not runs:
        return None

    best_candidate: EllipseSubshapeCandidate | None = None
    best_score = -1.0
    for start_index, run_length in runs:
        if run_length < max(8, len(contour.points) // 12):
            continue
        span = [contour.points[(start_index + offset) % len(contour.points)] for offset in range(run_length)]
        span_min_x = min(point.x for point in span)
        span_max_x = max(point.x for point in span)
        span_width = span_max_x - span_min_x
        if span_width < width * config.ellipse_subshape_min_span_ratio:
            continue

        axis_x = 0.5 * (span_min_x + span_max_x)
        mirrored_points = _mirror_span_points(span, axis_x)
        fit = fit_rotated_ellipse(mirrored_points)
        if fit is None:
            continue

        primitive = _symmetry_normalize_ellipse(fit.primitive, axis_x, config)
        aspect_ratio = max(primitive.radius_x, primitive.radius_y) / max(1e-6, min(primitive.radius_x, primitive.radius_y))
        if not (config.ellipse_subshape_min_aspect_ratio <= aspect_ratio <= config.ellipse_subshape_max_aspect_ratio):
            continue

        symmetry_offset = abs(primitive.center.x - axis_x) / max(1.0, width * 0.5)
        endpoint_balance = abs(span[0].y - span[-1].y) / max(1.0, primitive.radius_y)
        rotation_error = _horizontal_rotation_error(primitive.rotation_degrees) / max(
            1e-6, config.ellipse_subshape_rotation_tolerance_deg
        )
        confidence = _combine_score(
            _scaled_inverse(fit.normalized_error, config.ellipse_fit_tolerance * 1.75),
            _scaled_inverse(symmetry_offset, config.ellipse_subshape_symmetry_tolerance),
            _scaled_inverse(endpoint_balance, 0.7),
            min(1.0, span_width / max(1.0, width * 0.55)),
            _scaled_inverse(rotation_error, 1.0),
        )
        if confidence < config.ellipse_subshape_confidence_threshold:
            continue

        segment = _build_elliptical_arc_segment(span, primitive)
        if segment is None:
            continue

        candidate = EllipseSubshapeCandidate(
            start_source_index=start_index,
            end_source_index=(start_index + run_length - 1) % len(contour.points),
            start_anchor_index=-1,
            end_anchor_index=-1,
            segment=segment,
            confidence=confidence,
            symmetry_axis_x=axis_x,
            notes="Bottom ellipse-like subshape",
        )
        if confidence > best_score:
            best_candidate = candidate
            best_score = confidence

    return best_candidate


def attach_ellipse_subshape_anchors(
    candidate: EllipseSubshapeCandidate | None,
    anchor_indices: list[int],
) -> EllipseSubshapeCandidate | None:
    if candidate is None:
        return None
    if candidate.start_source_index not in anchor_indices or candidate.end_source_index not in anchor_indices:
        return None
    return replace(
        candidate,
        start_anchor_index=anchor_indices.index(candidate.start_source_index),
        end_anchor_index=anchor_indices.index(candidate.end_source_index),
    )


def required_anchor_indices(candidate: EllipseSubshapeCandidate | None) -> list[int]:
    if candidate is None:
        return []
    return [candidate.start_source_index, candidate.end_source_index]


def _contiguous_runs(flags: list[bool]) -> list[tuple[int, int]]:
    if not flags:
        return []
    runs: list[tuple[int, int]] = []
    count = len(flags)
    for index in range(count):
        if flags[index] and not flags[(index - 1) % count]:
            length = 0
            while flags[(index + length) % count] and length < count:
                length += 1
            runs.append((index, length))
    return runs


def _mirror_span_points(points: list[Point], axis_x: float) -> list[Point]:
    if len(points) < 3:
        return points[:]
    mirrored = [Point(2.0 * axis_x - point.x, point.y) for point in reversed(points[1:-1])]
    return points + mirrored


def _symmetry_normalize_ellipse(
    ellipse: PrimitiveEllipse,
    axis_x: float,
    config: HybridVectorizerV2Config,
) -> PrimitiveEllipse:
    rotation = ellipse.rotation_degrees
    if _horizontal_rotation_error(rotation) <= config.ellipse_subshape_rotation_tolerance_deg:
        rotation = 0.0
    return PrimitiveEllipse(
        center=Point(axis_x, ellipse.center.y),
        radius_x=ellipse.radius_x,
        radius_y=ellipse.radius_y,
        rotation_degrees=rotation,
    )


def _build_elliptical_arc_segment(points: list[Point], ellipse: PrimitiveEllipse) -> SegmentArcElliptical | None:
    if len(points) < 3:
        return None
    local = _to_ellipse_local(points, ellipse)
    rx = max(1e-6, ellipse.radius_x)
    ry = max(1e-6, ellipse.radius_y)
    angles = np.unwrap(np.arctan2(local[:, 1] / ry, local[:, 0] / rx))
    sweep_radians = float(angles[-1] - angles[0])
    if abs(sweep_radians) <= 0.2:
        return None
    return SegmentArcElliptical(
        start=points[0],
        end=points[-1],
        radius_x=ellipse.radius_x,
        radius_y=ellipse.radius_y,
        rotation_degrees=ellipse.rotation_degrees,
        large_arc=abs(sweep_radians) > np.pi,
        sweep=sweep_radians > 0.0,
    )


def _to_ellipse_local(points: list[Point], ellipse: PrimitiveEllipse) -> np.ndarray:
    angle = radians(ellipse.rotation_degrees)
    cos_a = cos(angle)
    sin_a = sin(angle)
    coords = np.asarray([(point.x - ellipse.center.x, point.y - ellipse.center.y) for point in points], dtype=np.float64)
    x_local = coords[:, 0] * cos_a + coords[:, 1] * sin_a
    y_local = -coords[:, 0] * sin_a + coords[:, 1] * cos_a
    return np.column_stack([x_local, y_local])


def _horizontal_rotation_error(rotation_degrees: float) -> float:
    normalized = abs(rotation_degrees) % 180.0
    return min(normalized, abs(180.0 - normalized))


def _scaled_inverse(value: float, limit: float) -> float:
    if limit <= 1e-6:
        return 1.0
    return max(0.0, min(1.0, 1.0 - value / limit))


def _combine_score(*scores: float) -> float:
    if not scores:
        return 0.0
    return float(sum(max(0.0, min(1.0, score)) for score in scores) / len(scores))
