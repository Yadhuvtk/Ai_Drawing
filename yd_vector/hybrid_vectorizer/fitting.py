from __future__ import annotations

import numpy as np

from yd_vector.hybrid_vectorizer.cleanup import douglas_peucker_closed, merge_near_duplicate_points
from yd_vector.hybrid_vectorizer.config import HybridVectorizerConfig
from yd_vector.hybrid_vectorizer.corner_modeling import CornerCandidate, classify_contour_corners
from yd_vector.hybrid_vectorizer.cutout_assembly import assemble_shape
from yd_vector.hybrid_vectorizer.geometry import (
    ClosedContour,
    ContourRegion,
    Loop,
    Point,
    Segment,
    SegmentArcCircular,
    SegmentArcElliptical,
    SegmentBezierCubic,
    SegmentBezierQuadratic,
    SegmentLine,
    Shape,
)
from yd_vector.hybrid_vectorizer.loop_builder import (
    build_circle_loop,
    build_ellipse_loop,
    build_polyline_loop,
    build_rectangle_loop,
    build_rounded_rectangle_loop,
)
from yd_vector.hybrid_vectorizer.shape_analysis import (
    CircleFitResult,
    CircularArcFitResult,
    EllipseFitResult,
    EllipticalArcFitResult,
    RectangleFitResult,
    RoundedRectangleFitResult,
    fit_circle,
    fit_circular_arc,
    fit_elliptical_arc,
    fit_rectangle,
    fit_rotated_ellipse,
    fit_rounded_rectangle,
)
from yd_vector.hybrid_vectorizer.topology_guard import validate_loop_against_contour, validate_shape_topology


def fit_region(
    region: ContourRegion,
    config: HybridVectorizerConfig,
    fill_color: str | None = None,
    stroke_color: str | None = None,
    shape_id: str | None = None,
    layer_id: str | None = None,
    z_index: int = 0,
) -> Shape:
    outer_loop = fit_contour(region.outer, config, polarity="positive")
    negative_loops = [fit_contour(hole, config, polarity="negative") for hole in region.holes]
    shape = assemble_shape(
        shape_id=shape_id or region.region_id,
        outer_loop=outer_loop,
        negative_loops=negative_loops,
        fill=fill_color or config.fill_color,
        stroke=stroke_color if stroke_color is not None else config.stroke_color,
        layer_id=layer_id,
        z_index=z_index,
    )
    if validate_shape_topology(shape, region.outer, region.holes, config):
        return shape

    fallback = assemble_shape(
        shape_id=shape_id or region.region_id,
        outer_loop=_polyline_loop_from_contour(region.outer, polarity="positive"),
        negative_loops=[_polyline_loop_from_contour(hole, polarity="negative") for hole in region.holes],
        fill=fill_color or config.fill_color,
        stroke=stroke_color if stroke_color is not None else config.stroke_color,
        layer_id=layer_id,
        z_index=z_index,
    )
    return fallback


def fit_contour(contour: ClosedContour, config: HybridVectorizerConfig, polarity: str = "positive") -> Loop:
    parameterized = _fit_parameterized_loop(contour, config, polarity)
    if parameterized is not None and validate_loop_against_contour(parameterized, contour, config):
        return parameterized

    segments = _fit_curve_segments(contour.points, config)
    candidate = Loop(
        loop_id=contour.contour_id,
        segments=segments,
        polarity=polarity,
        closed=True,
        source_contour_id=contour.contour_id,
        confidence=0.0,
    )
    if validate_loop_against_contour(candidate, contour, config):
        return candidate

    smooth_fallback = _fit_safe_spline_loop(contour, config, polarity)
    if smooth_fallback is not None and validate_loop_against_contour(smooth_fallback, contour, config):
        return smooth_fallback
    return _polyline_loop_from_contour(contour, polarity=polarity)


def _fit_parameterized_loop(contour: ClosedContour, config: HybridVectorizerConfig, polarity: str) -> Loop | None:
    candidates: list[tuple[float, Loop]] = []

    circle_result = fit_circle(contour.points)
    if _is_circle_candidate(circle_result, contour, config):
        candidates.append(
            (
                circle_result.confidence + 0.04,
                build_circle_loop(
                    loop_id=contour.contour_id,
                    circle=circle_result.primitive,
                    polarity=polarity,
                    source_contour_id=contour.contour_id,
                    confidence=circle_result.confidence,
                ),
            )
        )

    ellipse_result = fit_rotated_ellipse(contour.points)
    if _is_ellipse_candidate(ellipse_result, config):
        candidates.append(
            (
                ellipse_result.confidence + 0.02,
                build_ellipse_loop(
                    loop_id=contour.contour_id,
                    ellipse=ellipse_result.primitive,
                    polarity=polarity,
                    source_contour_id=contour.contour_id,
                    confidence=ellipse_result.confidence,
                ),
            )
        )

    rounded_rect_result = fit_rounded_rectangle(contour.points)
    if _is_rounded_rectangle_candidate(rounded_rect_result, config):
        candidates.append(
            (
                rounded_rect_result.confidence + 0.01,
                build_rounded_rectangle_loop(
                    loop_id=contour.contour_id,
                    rounded_rectangle=rounded_rect_result.primitive,
                    polarity=polarity,
                    source_contour_id=contour.contour_id,
                    confidence=rounded_rect_result.confidence,
                ),
            )
        )

    rectangle_result = fit_rectangle(contour.points)
    if _is_rectangle_candidate(rectangle_result, config):
        candidates.append(
            (
                rectangle_result.confidence,
                build_rectangle_loop(
                    loop_id=contour.contour_id,
                    rectangle=rectangle_result.primitive,
                    polarity=polarity,
                    source_contour_id=contour.contour_id,
                    confidence=rectangle_result.confidence,
                ),
            )
        )

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _is_circle_candidate(
    result: CircleFitResult | None,
    contour: ClosedContour,
    config: HybridVectorizerConfig,
) -> bool:
    if result is None:
        return False

    bbox = contour.bbox
    aspect_ratio = bbox.width / max(1e-6, bbox.height)
    return (
        0.85 <= aspect_ratio <= 1.15
        and result.radial_error <= config.circle_fit_tolerance
        and result.circularity >= config.circle_circularity_min
        and 0.75 <= result.area_ratio <= 1.25
        and result.confidence >= config.circle_confidence_threshold
    )


def _is_ellipse_candidate(result: EllipseFitResult | None, config: HybridVectorizerConfig) -> bool:
    if result is None:
        return False
    return (
        result.normalized_error <= config.ellipse_fit_tolerance
        and 0.65 <= result.area_ratio <= 1.35
        and result.confidence >= config.ellipse_confidence_threshold
    )


def _is_rectangle_candidate(result: RectangleFitResult | None, config: HybridVectorizerConfig) -> bool:
    if result is None:
        return False
    return (
        result.mean_error <= config.primitive_fit_error_threshold
        and 0.78 <= result.area_ratio <= 1.22
        and result.confidence >= config.rectangle_confidence_threshold
    )


def _is_rounded_rectangle_candidate(result: RoundedRectangleFitResult | None, config: HybridVectorizerConfig) -> bool:
    if result is None:
        return False

    min_dimension = min(result.primitive.width, result.primitive.height)
    radius_ratio = result.primitive.corner_radius / max(1e-6, min_dimension)
    return (
        result.mean_error <= config.primitive_fit_error_threshold
        and 0.72 <= result.area_ratio <= 1.28
        and 0.06 <= radius_ratio <= 0.30
        and result.confidence >= config.rounded_rectangle_confidence_threshold
    )


def _fit_curve_segments(points: list[Point], config: HybridVectorizerConfig) -> list[Segment]:
    if len(points) < 2:
        return []

    corner_candidates = classify_contour_corners(points, config)
    anchors = _build_anchor_indices(points, corner_candidates)
    segments: list[Segment] = []

    for anchor_index, start_index in enumerate(anchors):
        end_index = anchors[(anchor_index + 1) % len(anchors)]
        start_corner = corner_candidates.get(start_index)
        end_corner = corner_candidates.get(end_index)
        span = _build_trimmed_span(points, start_index, end_index, start_corner, end_corner)
        if len(span) >= 2:
            segments.extend(_fit_open_span(span, config))

        fillet = _corner_fillet_segment(end_corner)
        if fillet is not None:
            segments.append(fillet)
    return _merge_adjacent_lines(segments)


def _build_anchor_indices(points: list[Point], corner_candidates: dict[int, CornerCandidate]) -> list[int]:
    count = len(points)
    if count < 2:
        return [0]

    corner_indices = sorted(
        index
        for index, candidate in corner_candidates.items()
        if candidate.classification != "treat_as_smooth_curve"
    )

    if not corner_indices:
        step = max(1, count // 4)
        anchors = sorted({0, step % count, (2 * step) % count, (3 * step) % count})
        return anchors if len(anchors) >= 2 else [0, count // 2]

    if len(corner_indices) == 1:
        return sorted({corner_indices[0], (corner_indices[0] + count // 2) % count})
    return corner_indices


def _closed_span(points: list[Point], start_index: int, end_index: int) -> list[Point]:
    if start_index <= end_index:
        return points[start_index : end_index + 1]
    return points[start_index:] + points[: end_index + 1]


def _build_trimmed_span(
    points: list[Point],
    start_index: int,
    end_index: int,
    start_corner: CornerCandidate | None,
    end_corner: CornerCandidate | None,
) -> list[Point]:
    span = _closed_span(points, start_index, end_index)
    if not span:
        return []

    start_point = _corner_exit_point(start_corner) or span[0]
    end_point = _corner_entry_point(end_corner) or span[-1]

    trimmed: list[Point] = [start_point]
    for point in span[1:-1]:
        if _distance(trimmed[-1], point) > 1e-6:
            trimmed.append(point)
    if _distance(trimmed[-1], end_point) > 1e-6:
        trimmed.append(end_point)
    return trimmed if len(trimmed) >= 2 else [start_point, end_point]


def _corner_entry_point(candidate: CornerCandidate | None) -> Point | None:
    if candidate is None or candidate.classification != "apply_small_fillet" or candidate.fillet is None:
        return None
    return candidate.fillet.entry_point


def _corner_exit_point(candidate: CornerCandidate | None) -> Point | None:
    if candidate is None or candidate.classification != "apply_small_fillet" or candidate.fillet is None:
        return None
    return candidate.fillet.exit_point


def _corner_fillet_segment(candidate: CornerCandidate | None) -> Segment | None:
    if candidate is None or candidate.classification != "apply_small_fillet" or candidate.fillet is None:
        return None
    return candidate.fillet.as_arc()


def _fit_open_span(points: list[Point], config: HybridVectorizerConfig) -> list[Segment]:
    if len(points) < 2 or _distance(points[0], points[-1]) <= 1e-6:
        return []

    if len(points) <= 2:
        return [SegmentLine(start=points[0], end=points[-1])]

    if _line_fit_error(points) <= config.line_fit_tolerance:
        return [SegmentLine(start=points[0], end=points[-1])]

    arc_segment = _fit_arc_span(points, config)
    if arc_segment is not None:
        return [arc_segment]

    quadratic, quadratic_error, quadratic_split = _fit_quadratic_bezier(points)
    if quadratic is not None and quadratic_error <= config.quadratic_fit_tolerance:
        return [quadratic]

    cubic, cubic_error, cubic_split = _fit_cubic_bezier(points)
    if cubic is not None and cubic_error <= config.bezier_fit_tolerance:
        return [cubic]

    split_index = quadratic_split if quadratic_error <= cubic_error else cubic_split
    if split_index <= 1 or split_index >= len(points) - 1:
        return _fit_catmull_rom_chain(points)

    left = _fit_open_span(points[: split_index + 1], config)
    right = _fit_open_span(points[split_index:], config)
    return left + right


def _fit_quadratic_bezier(points: list[Point]) -> tuple[SegmentBezierQuadratic | None, float, int]:
    if len(points) < 3:
        return None, float("inf"), -1

    ts = _chord_length_parameterize(points)
    p0 = points[0]
    p2 = points[-1]
    denom = 0.0
    accum_x = 0.0
    accum_y = 0.0
    for point, t in zip(points[1:-1], ts[1:-1]):
        basis = 2.0 * (1.0 - t) * t
        if basis <= 1e-6:
            continue
        base_x = (1.0 - t) * (1.0 - t) * p0.x + t * t * p2.x
        base_y = (1.0 - t) * (1.0 - t) * p0.y + t * t * p2.y
        accum_x += basis * (point.x - base_x)
        accum_y += basis * (point.y - base_y)
        denom += basis * basis

    if denom <= 1e-8:
        return None, float("inf"), -1

    control = Point(accum_x / denom, accum_y / denom)
    bezier = SegmentBezierQuadratic(start=p0, control=control, end=p2)

    max_error = 0.0
    split_index = len(points) // 2
    for index, (point, t) in enumerate(zip(points, ts)):
        sample = _evaluate_quadratic_bezier(bezier, t)
        error = _distance(point, sample)
        if error > max_error:
            max_error = error
            split_index = index
    return bezier, max_error, split_index


def _fit_cubic_bezier(points: list[Point]) -> tuple[SegmentBezierCubic | None, float, int]:
    if len(points) < 4:
        return None, float("inf"), -1

    ts = _chord_length_parameterize(points)
    p0 = points[0]
    p3 = points[-1]
    basis_rows = []
    rhs_x = []
    rhs_y = []
    for point, t in zip(points, ts):
        omt = 1.0 - t
        b0 = omt * omt * omt
        b1 = 3.0 * omt * omt * t
        b2 = 3.0 * omt * t * t
        b3 = t * t * t
        basis_rows.append([b1, b2])
        rhs_x.append(point.x - (b0 * p0.x + b3 * p3.x))
        rhs_y.append(point.y - (b0 * p0.y + b3 * p3.y))

    matrix = np.asarray(basis_rows, dtype=np.float64)
    try:
        ctrl_x, *_ = np.linalg.lstsq(matrix, np.asarray(rhs_x, dtype=np.float64), rcond=None)
        ctrl_y, *_ = np.linalg.lstsq(matrix, np.asarray(rhs_y, dtype=np.float64), rcond=None)
    except np.linalg.LinAlgError:
        return None, float("inf"), -1

    bezier = SegmentBezierCubic(
        start=p0,
        control1=Point(float(ctrl_x[0]), float(ctrl_y[0])),
        control2=Point(float(ctrl_x[1]), float(ctrl_y[1])),
        end=p3,
    )

    max_error = 0.0
    split_index = len(points) // 2
    for index, (point, t) in enumerate(zip(points, ts)):
        sample = _evaluate_cubic_bezier(bezier, t)
        error = _distance(point, sample)
        if error > max_error:
            max_error = error
            split_index = index
    return bezier, max_error, split_index


def _fit_catmull_rom_chain(points: list[Point]) -> list[Segment]:
    if len(points) <= 2:
        return [SegmentLine(start=points[0], end=points[-1])]

    segments: list[Segment] = []
    for index in range(len(points) - 1):
        p0 = points[index - 1] if index > 0 else points[index]
        p1 = points[index]
        p2 = points[index + 1]
        p3 = points[index + 2] if index + 2 < len(points) else points[index + 1]
        c1 = Point(p1.x + (p2.x - p0.x) / 6.0, p1.y + (p2.y - p0.y) / 6.0)
        c2 = Point(p2.x - (p3.x - p1.x) / 6.0, p2.y - (p3.y - p1.y) / 6.0)
        segments.append(SegmentBezierCubic(start=p1, control1=c1, control2=c2, end=p2))
    return segments


def _fit_arc_span(points: list[Point], config: HybridVectorizerConfig) -> Segment | None:
    candidates: list[tuple[float, Segment]] = []

    circular_result = fit_circular_arc(points)
    if _is_circular_arc_candidate(circular_result, config):
        candidates.append((circular_result.confidence - circular_result.normalized_max_error, circular_result.segment))

    elliptical_result = fit_elliptical_arc(points)
    if _is_elliptical_arc_candidate(elliptical_result, config):
        candidates.append((elliptical_result.confidence - elliptical_result.normalized_max_error, elliptical_result.segment))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _is_circular_arc_candidate(result: CircularArcFitResult | None, config: HybridVectorizerConfig) -> bool:
    if result is None:
        return False
    return (
        result.normalized_max_error <= config.arc_fit_tolerance
        and result.sweep_degrees >= config.min_arc_sweep_degrees
        and result.confidence >= config.arc_confidence_threshold
    )


def _is_elliptical_arc_candidate(result: EllipticalArcFitResult | None, config: HybridVectorizerConfig) -> bool:
    if result is None:
        return False
    return (
        result.normalized_max_error <= max(config.arc_fit_tolerance, config.ellipse_fit_tolerance)
        and result.sweep_degrees >= config.min_arc_sweep_degrees
        and result.confidence >= config.arc_confidence_threshold
    )


def _line_fit_error(points: list[Point]) -> float:
    start = points[0]
    end = points[-1]
    return max(_point_to_segment_distance(point, start, end) for point in points[1:-1]) if len(points) > 2 else 0.0


def _point_to_segment_distance(point: Point, start: Point, end: Point) -> float:
    dx = end.x - start.x
    dy = end.y - start.y
    if dx == 0.0 and dy == 0.0:
        return _distance(point, start)

    t = ((point.x - start.x) * dx + (point.y - start.y) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    projection = Point(start.x + t * dx, start.y + t * dy)
    return _distance(point, projection)


def _chord_length_parameterize(points: list[Point]) -> list[float]:
    distances = [0.0]
    total = 0.0
    for index in range(1, len(points)):
        total += _distance(points[index - 1], points[index])
        distances.append(total)
    if total <= 0.0:
        return [0.0 for _ in points]
    return [value / total for value in distances]


def _evaluate_quadratic_bezier(bezier: SegmentBezierQuadratic, t: float) -> Point:
    omt = 1.0 - t
    x = omt * omt * bezier.start.x + 2.0 * omt * t * bezier.control.x + t * t * bezier.end.x
    y = omt * omt * bezier.start.y + 2.0 * omt * t * bezier.control.y + t * t * bezier.end.y
    return Point(x, y)


def _evaluate_cubic_bezier(bezier: SegmentBezierCubic, t: float) -> Point:
    omt = 1.0 - t
    x = (
        omt * omt * omt * bezier.start.x
        + 3.0 * omt * omt * t * bezier.control1.x
        + 3.0 * omt * t * t * bezier.control2.x
        + t * t * t * bezier.end.x
    )
    y = (
        omt * omt * omt * bezier.start.y
        + 3.0 * omt * omt * t * bezier.control1.y
        + 3.0 * omt * t * t * bezier.control2.y
        + t * t * t * bezier.end.y
    )
    return Point(x, y)


def _merge_adjacent_lines(segments: list[Segment]) -> list[Segment]:
    if not segments:
        return segments

    merged: list[Segment] = [segments[0]]
    for segment in segments[1:]:
        prev = merged[-1]
        if isinstance(prev, SegmentLine) and isinstance(segment, SegmentLine) and _are_lines_collinear(prev, segment):
            merged[-1] = SegmentLine(start=prev.start, end=segment.end)
            continue
        merged.append(segment)
    return merged


def _are_lines_collinear(left: SegmentLine, right: SegmentLine) -> bool:
    if left.end != right.start:
        return False
    tolerance = 0.25
    return _point_to_segment_distance(left.end, left.start, right.end) <= tolerance


def _distance(a: Point, b: Point) -> float:
    return float(np.hypot(a.x - b.x, a.y - b.y))


def _fit_safe_spline_loop(contour: ClosedContour, config: HybridVectorizerConfig, polarity: str) -> Loop | None:
    reduced = douglas_peucker_closed(contour.points, tolerance=max(0.35, config.simplify_tolerance * 0.45))
    reduced = merge_near_duplicate_points(reduced, merge_distance=max(0.01, config.merge_distance * 0.5))
    if len(reduced) < 4:
        return None

    corner_candidates = classify_contour_corners(reduced, config)
    sharp_corners = {
        index
        for index, candidate in corner_candidates.items()
        if candidate.classification == "preserve_sharp"
    }

    segments: list[Segment] = []
    count = len(reduced)
    tension = 0.12
    for index in range(count):
        start = reduced[index]
        end = reduced[(index + 1) % count]
        if index in sharp_corners or (index + 1) % count in sharp_corners:
            segments.append(SegmentLine(start=start, end=end))
            continue

        prev_point = reduced[index - 1]
        next_point = reduced[(index + 2) % count]
        control1 = Point(
            start.x + (end.x - prev_point.x) * tension,
            start.y + (end.y - prev_point.y) * tension,
        )
        control2 = Point(
            end.x - (next_point.x - start.x) * tension,
            end.y - (next_point.y - start.y) * tension,
        )
        segments.append(
            SegmentBezierCubic(
                start=start,
                control1=control1,
                control2=control2,
                end=end,
            )
        )

    return Loop(
        loop_id=contour.contour_id,
        segments=_merge_adjacent_lines(segments),
        polarity=polarity,
        closed=True,
        source_contour_id=contour.contour_id,
        confidence=0.0,
    )


def _polyline_loop_from_contour(contour: ClosedContour, polarity: str) -> Loop:
    return build_polyline_loop(
        loop_id=contour.contour_id,
        points=contour.points,
        polarity=polarity,
        source_contour_id=contour.contour_id,
        confidence=0.0,
    )
