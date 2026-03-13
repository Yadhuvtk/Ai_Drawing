from __future__ import annotations

from math import acos, hypot

from yd_vector.hybrid_vectorizer.geometry import Loop, Point, SegmentBezierCubic, SegmentLine
from yd_vector.hybrid_vectorizer.loop_builder import build_polyline_loop
from yd_vector.hybrid_vectorizer.shape_analysis import fit_circular_arc, fit_elliptical_arc
from yd_vector.hybrid_vectorizer_v2.config import HybridVectorizerV2Config
from yd_vector.hybrid_vectorizer_v2.geometry import ContourPartPlan


def fit_freeform_loop(plan: ContourPartPlan, config: HybridVectorizerV2Config) -> Loop:
    if plan.ellipse_subshape is not None:
        loop = _fit_loop_with_ellipse_subshape(plan, config)
        if loop is not None:
            return loop

    points = plan.anchor_points if len(plan.anchor_points) >= 3 else plan.contour.points
    if len(points) < 3:
        return build_polyline_loop(
            loop_id=plan.contour.contour_id,
            points=plan.contour.points,
            polarity=plan.polarity,
            source_contour_id=plan.contour.contour_id,
            confidence=0.0,
        )

    straight_edges = {(run.start_anchor, run.end_anchor) for run in plan.straight_runs}
    sharp_indices = {
        index
        for index in range(len(points))
        if _corner_angle(points, index) <= config.sharp_corner_angle_deg
    }

    segments = []
    for index in range(len(points)):
        next_index = (index + 1) % len(points)
        start = points[index]
        end = points[next_index]
        if _distance(start, end) <= 1e-6:
            continue

        span = _source_span(plan, index, next_index)
        if (index, next_index) in straight_edges or index in sharp_indices or next_index in sharp_indices:
            segments.append(SegmentLine(start=start, end=end))
            continue

        arc_segment = _fit_arc_segment(span, config)
        if arc_segment is not None:
            segments.append(arc_segment)
            continue

        prev_point = points[index - 1]
        next_point = points[(next_index + 1) % len(points)]
        segments.append(_catmull_rom_to_cubic(prev_point, start, end, next_point, config.cubic_tension))

    if not segments:
        return build_polyline_loop(
            loop_id=plan.contour.contour_id,
            points=plan.contour.points,
            polarity=plan.polarity,
            source_contour_id=plan.contour.contour_id,
            confidence=0.0,
        )

    return Loop(
        loop_id=plan.contour.contour_id,
        segments=segments,
        polarity=plan.polarity,
        closed=True,
        source_contour_id=plan.contour.contour_id,
        confidence=0.0,
    )


def _fit_loop_with_ellipse_subshape(plan: ContourPartPlan, config: HybridVectorizerV2Config) -> Loop | None:
    points = plan.anchor_points if len(plan.anchor_points) >= 3 else plan.contour.points
    ellipse_subshape = plan.ellipse_subshape
    if ellipse_subshape is None or len(points) < 3:
        return None

    straight_edges = {(run.start_anchor, run.end_anchor) for run in plan.straight_runs}
    sharp_indices = {
        index
        for index in range(len(points))
        if _corner_angle(points, index) <= config.sharp_corner_angle_deg
    }

    segments = []
    edge_count = len(points)
    processed_edges = 0
    index = 0
    while processed_edges < edge_count:
        if index == ellipse_subshape.start_anchor_index:
            segments.append(ellipse_subshape.segment)
            step_count = (ellipse_subshape.end_anchor_index - ellipse_subshape.start_anchor_index) % edge_count
            if step_count <= 0:
                step_count = edge_count
            processed_edges += step_count
            index = ellipse_subshape.end_anchor_index
            continue

        next_index = (index + 1) % edge_count
        start = points[index]
        end = points[next_index]
        if _distance(start, end) > 1e-6:
            span = _source_span(plan, index, next_index)
            if (index, next_index) in straight_edges or index in sharp_indices or next_index in sharp_indices:
                segments.append(SegmentLine(start=start, end=end))
            else:
                arc_segment = _fit_arc_segment(span, config)
                if arc_segment is not None:
                    segments.append(arc_segment)
                else:
                    prev_point = points[index - 1]
                    after_next = points[(next_index + 1) % edge_count]
                    segments.append(_catmull_rom_to_cubic(prev_point, start, end, after_next, config.cubic_tension))

        processed_edges += 1
        index = next_index

    if not segments:
        return None

    return Loop(
        loop_id=plan.contour.contour_id,
        segments=segments,
        polarity=plan.polarity,
        closed=True,
        source_contour_id=plan.contour.contour_id,
        confidence=ellipse_subshape.confidence,
    )


def _fit_arc_segment(points: list[Point], config: HybridVectorizerV2Config) -> object | None:
    if len(points) < 4:
        return None

    circular = fit_circular_arc(points)
    if (
        circular is not None
        and circular.normalized_max_error <= config.arc_fit_tolerance
        and circular.sweep_degrees >= config.min_arc_sweep_degrees
        and circular.confidence >= config.arc_confidence_threshold
    ):
        return circular.segment

    elliptical = fit_elliptical_arc(points)
    if (
        elliptical is not None
        and elliptical.normalized_max_error <= config.arc_fit_tolerance
        and elliptical.sweep_degrees >= config.min_arc_sweep_degrees
        and elliptical.confidence >= config.arc_confidence_threshold
    ):
        return elliptical.segment
    return None


def _source_span(plan: ContourPartPlan, start_anchor: int, end_anchor: int) -> list[Point]:
    source = plan.contour.points
    start_index = plan.anchor_indices[start_anchor]
    end_index = plan.anchor_indices[end_anchor]
    if start_index <= end_index:
        return source[start_index : end_index + 1]
    return source[start_index:] + source[: end_index + 1]


def _catmull_rom_to_cubic(
    prev_point: Point,
    start: Point,
    end: Point,
    next_point: Point,
    tension: float,
) -> SegmentBezierCubic:
    scale = max(0.0, min(1.0, tension)) / 6.0
    control1 = Point(
        start.x + (end.x - prev_point.x) * scale,
        start.y + (end.y - prev_point.y) * scale,
    )
    control2 = Point(
        end.x - (next_point.x - start.x) * scale,
        end.y - (next_point.y - start.y) * scale,
    )
    control1 = _clamp_handle(start, control1, end)
    control2 = _clamp_handle(end, control2, start)
    return SegmentBezierCubic(start=start, control1=control1, control2=control2, end=end)


def _clamp_handle(origin: Point, control: Point, other_end: Point) -> Point:
    max_length = _distance(origin, other_end) * 0.45
    handle_length = _distance(origin, control)
    if handle_length <= max_length or handle_length <= 1e-6:
        return control
    scale = max_length / handle_length
    return Point(
        origin.x + (control.x - origin.x) * scale,
        origin.y + (control.y - origin.y) * scale,
    )


def _corner_angle(points: list[Point], index: int) -> float:
    prev_point = points[index - 1]
    point = points[index]
    next_point = points[(index + 1) % len(points)]
    ux = prev_point.x - point.x
    uy = prev_point.y - point.y
    vx = next_point.x - point.x
    vy = next_point.y - point.y
    mag_u = max(1e-6, hypot(ux, uy))
    mag_v = max(1e-6, hypot(vx, vy))
    dot = max(-1.0, min(1.0, (ux * vx + uy * vy) / (mag_u * mag_v)))
    return acos(dot) * 180.0 / 3.141592653589793


def _distance(left: Point, right: Point) -> float:
    return hypot(left.x - right.x, left.y - right.y)
