from __future__ import annotations

from dataclasses import replace
from math import acos, ceil, degrees, exp, radians

from yd_vector.hybrid_vectorizer.config import HybridVectorizerConfig
from yd_vector.hybrid_vectorizer.geometry import (
    ClosedContour,
    ContourRegion,
    Point,
    bounding_box_from_points,
    distance,
    polygon_area,
)
from yd_vector.hybrid_vectorizer.shape_analysis import detect_narrow_gap_indices
from yd_vector.hybrid_vectorizer.topology_guard import validate_contour_points


def cleanup_region(region: ContourRegion, config: HybridVectorizerConfig) -> ContourRegion:
    outer = cleanup_contour(region.outer, config)
    holes = [cleanup_contour(hole, config) for hole in region.holes]
    outer.children_ids = [hole.contour_id for hole in holes]
    return ContourRegion(region_id=region.region_id, outer=outer, holes=holes)


def cleanup_contour(contour: ClosedContour, config: HybridVectorizerConfig) -> ClosedContour:
    original_points = merge_near_duplicate_points(contour.points, max(0.01, config.merge_distance * 0.5))
    points = gaussian_smooth_closed_contour(original_points, sigma=1.0)
    points = remove_collinear_points(points, tolerance=max(0.05, config.simplify_tolerance * 0.12))
    points = douglas_peucker_closed(points, tolerance=min(0.8, max(0.2, config.simplify_tolerance)))
    if not validate_contour_points(contour, points, config):
        points = original_points

    anchor_points = points[:]
    corner_protection_angle = max(config.corner_angle_threshold_degrees, config.min_corner_angle_deg)
    corner_indices = detect_corner_indices(anchor_points, corner_protection_angle)
    gap_indices = (
        detect_narrow_gap_indices(
            anchor_points,
            gap_distance=config.gap_preservation_distance,
            protect_span=config.gap_protection_span,
            min_index_separation=3,
        )
        if config.preserve_narrow_gaps
        else set()
    )
    smoothed = smooth_closed_contour(
        anchor_points,
        corner_indices=corner_indices | gap_indices,
        iterations=config.smooth_iterations,
        strength=config.smooth_strength,
    )
    smoothed = collapse_short_edges(
        smoothed,
        min_length=max(0.4, config.merge_distance * 0.9),
        protected_indices=corner_indices | gap_indices,
    )
    smoothed = simplify_closed_preserving_indices(
        smoothed,
        tolerance=max(0.2, config.simplify_tolerance * 0.45),
        protected_indices=corner_indices | gap_indices,
    )
    smoothed = remove_collinear_points(
        smoothed,
        tolerance=max(0.06, config.simplify_tolerance * 0.18),
        protected_indices=corner_indices | gap_indices,
    )
    if len(smoothed) < 3 or not validate_contour_points(contour, smoothed, config):
        points = anchor_points if validate_contour_points(contour, anchor_points, config) else original_points
    else:
        points = smoothed

    return replace(
        contour,
        points=points,
        area=abs(polygon_area(points)),
        bbox=bounding_box_from_points(points),
    )


def merge_near_duplicate_points(points: list[Point], merge_distance: float, closed: bool = True) -> list[Point]:
    if len(points) < 2:
        return points

    merged = [points[0]]
    for point in points[1:]:
        if distance(merged[-1], point) >= merge_distance:
            merged.append(point)

    if closed and len(merged) > 2 and distance(merged[0], merged[-1]) < merge_distance:
        merged.pop()
    return merged


def simplify_closed_contour(
    points: list[Point],
    tolerance: float,
    corner_angle_threshold_degrees: float,
) -> list[Point]:
    simplified = douglas_peucker_closed(points, tolerance=tolerance)
    corners = detect_corner_indices(simplified, corner_angle_threshold_degrees)
    return smooth_closed_contour(simplified, corner_indices=corners, iterations=1, strength=0.35)


def douglas_peucker_closed(points: list[Point], tolerance: float) -> list[Point]:
    if len(points) <= 3 or tolerance <= 0.0:
        return points

    start_index = min(range(len(points)), key=lambda idx: (points[idx].x, points[idx].y))
    opposite_index = max(
        range(len(points)),
        key=lambda idx: distance(points[start_index], points[idx]),
    )
    if start_index == opposite_index:
        return points

    chain_a = _slice_closed(points, start_index, opposite_index)
    chain_b = _slice_closed(points, opposite_index, start_index)
    simplified_a = _douglas_peucker_open(chain_a, tolerance)
    simplified_b = _douglas_peucker_open(chain_b, tolerance)
    combined = simplified_a[:-1] + simplified_b[:-1]
    return merge_near_duplicate_points(combined, merge_distance=max(0.01, tolerance * 0.1))


def douglas_peucker_open(points: list[Point], tolerance: float) -> list[Point]:
    if len(points) <= 2 or tolerance <= 0.0:
        return points
    return _douglas_peucker_open(points, tolerance)


def simplify_closed_preserving_indices(
    points: list[Point],
    tolerance: float,
    protected_indices: set[int] | None = None,
) -> list[Point]:
    if len(points) <= 3 or tolerance <= 0.0:
        return points

    protected = {index % len(points) for index in (protected_indices or set())}
    if not protected:
        return douglas_peucker_closed(points, tolerance)

    if len(protected) == 1:
        protected.add((next(iter(protected)) + len(points) // 2) % len(points))

    ordered = sorted(protected)
    simplified: list[Point] = []
    for offset, start_index in enumerate(ordered):
        end_index = ordered[(offset + 1) % len(ordered)]
        span = _slice_closed(points, start_index, end_index)
        reduced_span = _douglas_peucker_open(span, tolerance)
        if not simplified:
            simplified.extend(reduced_span)
        else:
            simplified.extend(reduced_span[1:])
    return merge_near_duplicate_points(simplified, merge_distance=max(0.01, tolerance * 0.1))


def gaussian_smooth_closed_contour(points: list[Point], sigma: float) -> list[Point]:
    if len(points) < 5 or sigma <= 0.0:
        return points

    radius = max(1, int(ceil(3.0 * sigma)))
    kernel = _gaussian_kernel(radius, sigma)
    smoothed: list[Point] = []
    count = len(points)
    for index in range(count):
        accum_x = 0.0
        accum_y = 0.0
        for offset, weight in zip(range(-radius, radius + 1), kernel):
            sample = points[(index + offset) % count]
            accum_x += sample.x * weight
            accum_y += sample.y * weight
        smoothed.append(Point(accum_x, accum_y))
    return smoothed


def detect_corner_indices(points: list[Point], corner_angle_threshold_degrees: float) -> set[int]:
    if len(points) < 3:
        return set()

    corners = set()
    for index, point in enumerate(points):
        prev_point = points[index - 1]
        next_point = points[(index + 1) % len(points)]
        angle = _corner_angle_degrees(prev_point, point, next_point)
        if angle <= corner_angle_threshold_degrees:
            corners.add(index)
    return corners


def smooth_closed_contour(
    points: list[Point],
    corner_indices: set[int],
    iterations: int,
    strength: float,
) -> list[Point]:
    if len(points) < 5 or iterations <= 0 or strength <= 0.0:
        return points

    if _curvature_variance(points) < 0.3:
        iterations = max(iterations, 3)

    strength = max(0.0, min(0.85, strength))
    current = points[:]
    protected_indices = _expand_protected_indices(corner_indices, len(points), radius=1)
    for _ in range(iterations):
        current = _smooth_closed_pass(current, protected_indices, strength)
        current = _smooth_closed_pass(current, protected_indices, -0.45 * strength)
    return current


def remove_collinear_points(
    points: list[Point],
    tolerance: float,
    protected_indices: set[int] | None = None,
) -> list[Point]:
    if len(points) <= 3:
        return points

    protected_indices = protected_indices or set()
    reduced: list[Point] = []
    for index, point in enumerate(points):
        if index in protected_indices:
            reduced.append(point)
            continue

        prev_point = points[index - 1]
        next_point = points[(index + 1) % len(points)]
        deviation = _point_to_segment_distance(point, prev_point, next_point)
        if deviation > tolerance:
            reduced.append(point)
    return reduced if len(reduced) >= 3 else points


def collapse_short_edges(
    points: list[Point],
    min_length: float,
    protected_indices: set[int] | None = None,
) -> list[Point]:
    if len(points) <= 3 or min_length <= 0.0:
        return points

    protected_indices = protected_indices or set()
    current = points[:]
    changed = True
    while changed and len(current) > 3:
        changed = False
        count = len(current)
        for index in range(count):
            next_index = (index + 1) % count
            if index in protected_indices or next_index in protected_indices:
                continue
            if distance(current[index], current[next_index]) >= min_length:
                continue

            prev_point = current[index - 1]
            point = current[index]
            next_point = current[next_index]
            after_next = current[(next_index + 1) % count]
            keep_next = _corner_angle_degrees(prev_point, point, after_next) > _corner_angle_degrees(prev_point, next_point, after_next)
            remove_index = index if keep_next else next_index
            del current[remove_index]
            protected_indices = {
                idx - 1 if idx > remove_index else idx
                for idx in protected_indices
                if idx != remove_index
            }
            changed = True
            break
    return current


def _smooth_closed_pass(points: list[Point], protected_indices: set[int], strength: float) -> list[Point]:
    updated = points[:]
    count = len(points)
    for index, point in enumerate(points):
        if index in protected_indices:
            continue

        prev2 = points[(index - 2) % count]
        prev1 = points[index - 1]
        next1 = points[(index + 1) % count]
        next2 = points[(index + 2) % count]
        target = Point(
            x=(prev2.x + 4.0 * prev1.x + 6.0 * point.x + 4.0 * next1.x + next2.x) / 16.0,
            y=(prev2.y + 4.0 * prev1.y + 6.0 * point.y + 4.0 * next1.y + next2.y) / 16.0,
        )
        angle = _corner_angle_degrees(prev1, point, next1)
        curve_factor = _smooth_curve_factor(angle)
        neighbor_factor = 0.65 if ((index - 1) % count in protected_indices or (index + 1) % count in protected_indices) else 1.0
        local_strength = strength * curve_factor * neighbor_factor
        updated[index] = Point(
            x=point.x + local_strength * (target.x - point.x),
            y=point.y + local_strength * (target.y - point.y),
        )
    return updated


def _expand_protected_indices(indices: set[int], count: int, radius: int) -> set[int]:
    if not indices or count <= 0 or radius <= 0:
        return set(indices)
    expanded = set(indices)
    for index in indices:
        for offset in range(-radius, radius + 1):
            expanded.add((index + offset) % count)
    return expanded


def _smooth_curve_factor(angle_degrees: float) -> float:
    normalized = max(0.0, min(1.0, (angle_degrees - 105.0) / 65.0))
    return 0.2 + 0.8 * normalized


def _gaussian_kernel(radius: int, sigma: float) -> list[float]:
    weights = [exp(-((offset * offset) / max(1e-6, 2.0 * sigma * sigma))) for offset in range(-radius, radius + 1)]
    total = sum(weights)
    return [weight / total for weight in weights]


def _curvature_variance(points: list[Point]) -> float:
    if len(points) < 5:
        return float("inf")

    curvatures: list[float] = []
    for index, point in enumerate(points):
        prev_point = points[index - 1]
        next_point = points[(index + 1) % len(points)]
        angle = _corner_angle_degrees(prev_point, point, next_point)
        curvatures.append(max(0.0, radians(180.0 - angle)))

    mean = sum(curvatures) / len(curvatures)
    return sum((curvature - mean) ** 2 for curvature in curvatures) / len(curvatures)


def _slice_closed(points: list[Point], start_index: int, end_index: int) -> list[Point]:
    if start_index <= end_index:
        return points[start_index : end_index + 1]
    return points[start_index:] + points[: end_index + 1]


def _douglas_peucker_open(points: list[Point], tolerance: float) -> list[Point]:
    if len(points) <= 2:
        return points

    max_distance = -1.0
    max_index = -1
    start = points[0]
    end = points[-1]
    for index in range(1, len(points) - 1):
        deviation = _point_to_segment_distance(points[index], start, end)
        if deviation > max_distance:
            max_distance = deviation
            max_index = index

    if max_distance <= tolerance or max_index < 0:
        return [start, end]

    left = _douglas_peucker_open(points[: max_index + 1], tolerance)
    right = _douglas_peucker_open(points[max_index:], tolerance)
    return left[:-1] + right


def _point_to_segment_distance(point: Point, start: Point, end: Point) -> float:
    dx = end.x - start.x
    dy = end.y - start.y
    if dx == 0.0 and dy == 0.0:
        return distance(point, start)

    t = ((point.x - start.x) * dx + (point.y - start.y) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    projection = Point(start.x + t * dx, start.y + t * dy)
    return distance(point, projection)


def _corner_angle_degrees(prev_point: Point, point: Point, next_point: Point) -> float:
    ux = prev_point.x - point.x
    uy = prev_point.y - point.y
    vx = next_point.x - point.x
    vy = next_point.y - point.y

    mag_u = (ux * ux + uy * uy) ** 0.5
    mag_v = (vx * vx + vy * vy) ** 0.5
    if mag_u == 0.0 or mag_v == 0.0:
        return 180.0

    dot = (ux * vx + uy * vy) / (mag_u * mag_v)
    dot = max(-1.0, min(1.0, dot))
    return degrees(acos(dot))


# TODO: Add corner scoring and learned cleanup policies for harder raster inputs.
