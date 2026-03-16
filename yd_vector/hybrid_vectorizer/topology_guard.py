from __future__ import annotations

from math import acos, cos, degrees, radians, sin, sqrt

from yd_vector.hybrid_vectorizer.config import HybridVectorizerConfig
from yd_vector.hybrid_vectorizer.geometry import (
    BoundingBox,
    ClosedContour,
    Loop,
    Point,
    SegmentArcCircular,
    SegmentArcElliptical,
    SegmentBezierCubic,
    SegmentBezierQuadratic,
    SegmentLine,
    Shape,
    bounding_box_from_points,
    distance,
    polygon_area,
)


def validate_contour_points(
    source_contour: ClosedContour,
    candidate_points: list[Point],
    config: HybridVectorizerConfig,
) -> bool:
    if len(candidate_points) < 3:
        return False
    normalized = _dedupe_closed_points(candidate_points)
    if len(normalized) < 3:
        return False
    if polygon_self_intersects(normalized):
        return False

    source_area = max(1e-6, float(source_contour.area))
    candidate_area = abs(polygon_area(normalized))
    area_tolerance = config.hole_area_tolerance if source_contour.is_hole else config.topology_area_tolerance
    if abs(candidate_area - source_area) / source_area > area_tolerance:
        return False

    source_bbox = source_contour.bbox
    candidate_bbox = bounding_box_from_points(normalized)
    bbox_threshold = config.hole_bbox_iou_threshold if source_contour.is_hole else config.topology_bbox_iou_threshold
    if bbox_iou(source_bbox, candidate_bbox) < bbox_threshold:
        return False
    return True


def validate_loop_against_contour(
    loop: Loop,
    contour: ClosedContour,
    config: HybridVectorizerConfig,
) -> bool:
    sampled = sample_loop_points(loop)
    return validate_contour_points(contour, sampled, config)


def validate_shape_topology(shape: Shape, region_outer: ClosedContour, region_holes: list[ClosedContour], config: HybridVectorizerConfig) -> bool:
    outer_points = sample_loop_points(shape.outer_loop)
    if not validate_contour_points(region_outer, outer_points, config):
        return False

    outer_polygon = _dedupe_closed_points(outer_points)
    hole_map = {hole.contour_id: hole for hole in region_holes}
    validated_holes: list[list[Point]] = []
    for hole_loop in shape.negative_loops:
        source_hole = hole_map.get(hole_loop.source_contour_id or hole_loop.loop_id)
        if source_hole is None:
            return False
        hole_points = sample_loop_points(hole_loop)
        if not validate_contour_points(source_hole, hole_points, config):
            return False
        hole_polygon = _dedupe_closed_points(hole_points)
        centroid = polygon_centroid(hole_polygon)
        if centroid is None or not point_in_polygon(centroid, outer_polygon):
            return False
        if polygons_intersect(outer_polygon, hole_polygon):
            return False
        for other_hole in validated_holes:
            if polygons_intersect(other_hole, hole_polygon):
                return False
        validated_holes.append(hole_polygon)
    return True


def sample_loop_points(loop: Loop, samples_per_curve: int = 18) -> list[Point]:
    if not loop.segments:
        return []

    points: list[Point] = []
    for segment in loop.segments:
        sampled = sample_segment_points(segment, samples_per_curve=samples_per_curve)
        if not sampled:
            continue
        if points and distance(points[-1], sampled[0]) <= 1e-6:
            points.extend(sampled[1:])
        else:
            points.extend(sampled)
    return _dedupe_closed_points(points)


def sample_segment_points(segment: object, samples_per_curve: int = 12) -> list[Point]:
    if isinstance(segment, SegmentLine):
        return [segment.start, segment.end]
    if isinstance(segment, SegmentBezierQuadratic):
        return [_evaluate_quadratic(segment, t / samples_per_curve) for t in range(samples_per_curve + 1)]
    if isinstance(segment, SegmentBezierCubic):
        return [_evaluate_cubic(segment, t / samples_per_curve) for t in range(samples_per_curve + 1)]
    if isinstance(segment, SegmentArcCircular):
        return _sample_arc(
            segment.start,
            segment.end,
            segment.radius,
            segment.radius,
            0.0,
            segment.large_arc,
            segment.sweep,
            samples_per_curve,
        )
    if isinstance(segment, SegmentArcElliptical):
        return _sample_arc(
            segment.start,
            segment.end,
            segment.radius_x,
            segment.radius_y,
            segment.rotation_degrees,
            segment.large_arc,
            segment.sweep,
            samples_per_curve,
        )
    return []


def polygon_self_intersects(points: list[Point]) -> bool:
    if len(points) < 4:
        return False
    count = len(points)
    for left in range(count):
        left_start = points[left]
        left_end = points[(left + 1) % count]
        for right in range(left + 1, count):
            if abs(left - right) <= 1 or (left == 0 and right == count - 1):
                continue
            right_start = points[right]
            right_end = points[(right + 1) % count]
            if segments_intersect(left_start, left_end, right_start, right_end):
                return True
    return False


def polygons_intersect(left: list[Point], right: list[Point]) -> bool:
    for index in range(len(left)):
        left_start = left[index]
        left_end = left[(index + 1) % len(left)]
        for other_index in range(len(right)):
            right_start = right[other_index]
            right_end = right[(other_index + 1) % len(right)]
            if segments_intersect(left_start, left_end, right_start, right_end):
                return True
    return False


def point_in_polygon(point: Point, polygon: list[Point]) -> bool:
    winding_number = 0
    count = len(polygon)
    for index in range(count):
        a = polygon[index]
        b = polygon[(index + 1) % count]
        if _point_on_segment(point, a, b):
            return True

        if a.y <= point.y:
            if b.y > point.y and _is_left(a, b, point) > 0.0:
                winding_number += 1
        elif b.y <= point.y and _is_left(a, b, point) < 0.0:
            winding_number -= 1
    return winding_number != 0


def polygon_centroid(points: list[Point]) -> Point | None:
    if len(points) < 3:
        return None
    area = polygon_area(points)
    if abs(area) <= 1e-8:
        x = sum(point.x for point in points) / len(points)
        y = sum(point.y for point in points) / len(points)
        return Point(x, y)

    factor = 1.0 / (6.0 * area)
    accum_x = 0.0
    accum_y = 0.0
    for index, point in enumerate(points):
        nxt = points[(index + 1) % len(points)]
        cross = point.x * nxt.y - nxt.x * point.y
        accum_x += (point.x + nxt.x) * cross
        accum_y += (point.y + nxt.y) * cross
    return Point(accum_x * factor, accum_y * factor)


def bbox_iou(left: BoundingBox, right: BoundingBox) -> float:
    inter_min_x = max(left.min_x, right.min_x)
    inter_min_y = max(left.min_y, right.min_y)
    inter_max_x = min(left.max_x, right.max_x)
    inter_max_y = min(left.max_y, right.max_y)
    inter_w = max(0.0, inter_max_x - inter_min_x)
    inter_h = max(0.0, inter_max_y - inter_min_y)
    intersection = inter_w * inter_h
    left_area = max(0.0, left.width) * max(0.0, left.height)
    right_area = max(0.0, right.width) * max(0.0, right.height)
    union = left_area + right_area - intersection
    if union <= 1e-8:
        return 1.0
    return intersection / union


def segments_intersect(a0: Point, a1: Point, b0: Point, b1: Point) -> bool:
    if _shares_endpoint(a0, a1, b0, b1):
        return False
    o1 = _orientation(a0, a1, b0)
    o2 = _orientation(a0, a1, b1)
    o3 = _orientation(b0, b1, a0)
    o4 = _orientation(b0, b1, a1)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(a0, b0, a1):
        return True
    if o2 == 0 and _on_segment(a0, b1, a1):
        return True
    if o3 == 0 and _on_segment(b0, a0, b1):
        return True
    if o4 == 0 and _on_segment(b0, a1, b1):
        return True
    return False


def _evaluate_quadratic(segment: SegmentBezierQuadratic, t: float) -> Point:
    omt = 1.0 - t
    return Point(
        omt * omt * segment.start.x + 2.0 * omt * t * segment.control.x + t * t * segment.end.x,
        omt * omt * segment.start.y + 2.0 * omt * t * segment.control.y + t * t * segment.end.y,
    )


def _evaluate_cubic(segment: SegmentBezierCubic, t: float) -> Point:
    omt = 1.0 - t
    return Point(
        omt * omt * omt * segment.start.x
        + 3.0 * omt * omt * t * segment.control1.x
        + 3.0 * omt * t * t * segment.control2.x
        + t * t * t * segment.end.x,
        omt * omt * omt * segment.start.y
        + 3.0 * omt * omt * t * segment.control1.y
        + 3.0 * omt * t * t * segment.control2.y
        + t * t * t * segment.end.y,
    )


def _sample_arc(
    start: Point,
    end: Point,
    radius_x: float,
    radius_y: float,
    rotation_degrees: float,
    large_arc: bool,
    sweep: bool,
    samples_per_curve: int,
) -> list[Point]:
    if radius_x <= 1e-6 or radius_y <= 1e-6:
        return [start, end]

    phi = radians(rotation_degrees)
    cos_phi = cos(phi)
    sin_phi = sin(phi)
    dx = (start.x - end.x) * 0.5
    dy = (start.y - end.y) * 0.5
    x1p = cos_phi * dx + sin_phi * dy
    y1p = -sin_phi * dx + cos_phi * dy

    rx = abs(radius_x)
    ry = abs(radius_y)
    lambda_value = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry)
    if lambda_value > 1.0:
        scale = sqrt(lambda_value)
        rx *= scale
        ry *= scale

    numerator = max(
        0.0,
        (rx * rx * ry * ry) - (rx * rx * y1p * y1p) - (ry * ry * x1p * x1p),
    )
    denominator = max(1e-9, (rx * rx * y1p * y1p) + (ry * ry * x1p * x1p))
    sign = -1.0 if large_arc == sweep else 1.0
    coef = sign * sqrt(numerator / denominator) if numerator > 0.0 else 0.0
    cxp = coef * ((rx * y1p) / ry)
    cyp = coef * (-(ry * x1p) / rx)

    cx = cos_phi * cxp - sin_phi * cyp + (start.x + end.x) * 0.5
    cy = sin_phi * cxp + cos_phi * cyp + (start.y + end.y) * 0.5

    start_angle = _vector_angle(1.0, 0.0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    sweep_angle = _vector_angle(
        (x1p - cxp) / rx,
        (y1p - cyp) / ry,
        (-x1p - cxp) / rx,
        (-y1p - cyp) / ry,
    )
    if not sweep and sweep_angle > 0.0:
        sweep_angle -= 2.0 * 3.141592653589793
    elif sweep and sweep_angle < 0.0:
        sweep_angle += 2.0 * 3.141592653589793

    points: list[Point] = []
    for index in range(samples_per_curve + 1):
        t = index / samples_per_curve
        theta = start_angle + sweep_angle * t
        cos_theta = cos(theta)
        sin_theta = sin(theta)
        x = cos_phi * rx * cos_theta - sin_phi * ry * sin_theta + cx
        y = sin_phi * rx * cos_theta + cos_phi * ry * sin_theta + cy
        points.append(Point(x, y))
    return points


def _vector_angle(ux: float, uy: float, vx: float, vy: float) -> float:
    dot = ux * vx + uy * vy
    det = ux * vy - uy * vx
    mag = max(1e-9, sqrt(ux * ux + uy * uy) * sqrt(vx * vx + vy * vy))
    value = max(-1.0, min(1.0, dot / mag))
    angle = acos(value)
    return angle if det >= 0.0 else -angle


def _dedupe_closed_points(points: list[Point]) -> list[Point]:
    if not points:
        return []
    deduped = [points[0]]
    for point in points[1:]:
        if distance(deduped[-1], point) > 1e-6:
            deduped.append(point)
    if len(deduped) > 2 and distance(deduped[0], deduped[-1]) <= 1e-6:
        deduped.pop()
    return deduped


def _orientation(a: Point, b: Point, c: Point) -> int:
    value = (b.y - a.y) * (c.x - b.x) - (b.x - a.x) * (c.y - b.y)
    if abs(value) <= 1e-8:
        return 0
    return 1 if value > 0.0 else 2


def _on_segment(a: Point, b: Point, c: Point) -> bool:
    return (
        min(a.x, c.x) - 1e-8 <= b.x <= max(a.x, c.x) + 1e-8
        and min(a.y, c.y) - 1e-8 <= b.y <= max(a.y, c.y) + 1e-8
    )


def _shares_endpoint(a0: Point, a1: Point, b0: Point, b1: Point) -> bool:
    return (
        distance(a0, b0) <= 1e-6
        or distance(a0, b1) <= 1e-6
        or distance(a1, b0) <= 1e-6
        or distance(a1, b1) <= 1e-6
    )


def _is_left(a: Point, b: Point, point: Point) -> float:
    return (b.x - a.x) * (point.y - a.y) - (point.x - a.x) * (b.y - a.y)


def _point_on_segment(point: Point, start: Point, end: Point) -> bool:
    return abs(_is_left(start, end, point)) <= 1e-8 and _on_segment(start, point, end)
