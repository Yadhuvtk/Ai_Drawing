from __future__ import annotations

from dataclasses import replace
from math import acos, ceil, degrees, exp, radians

from yd_vector.hybrid_vectorizer.config import HybridVectorizerConfig
from yd_vector.hybrid_vectorizer.contour_extraction import RegionShape
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


HARD_CORNER_ANGLE_THRESHOLD_DEGREES = 60.0


def cleanup_region(region: ContourRegion, config: HybridVectorizerConfig) -> ContourRegion:
    outer = cleanup_contour(region.outer, config)
    holes = [cleanup_contour(hole, config) for hole in region.holes]
    outer.children_ids = [hole.contour_id for hole in holes]
    if isinstance(region, RegionShape):
        return RegionShape(
            region_id=region.region_id,
            outer=outer,
            holes=holes,
            fill_polarity=region.fill_polarity,
            metadata=region.metadata.copy(),
        )
    return ContourRegion(region_id=region.region_id, outer=outer, holes=holes)


def cleanup_contour(contour: ClosedContour, config: HybridVectorizerConfig) -> ClosedContour:
    original_points = merge_near_duplicate_points(contour.points, max(0.01, config.merge_distance * 0.5))
    hard_corner_indices = detect_hard_corner_indices(
        original_points,
        angle_threshold_deg=HARD_CORNER_ANGLE_THRESHOLD_DEGREES,
    )
    gap_indices = (
        detect_narrow_gap_indices(
            original_points,
            gap_distance=config.gap_preservation_distance,
            protect_span=config.gap_protection_span,
            min_index_separation=3,
        )
        if config.preserve_narrow_gaps
        else set()
    )
    structural_indices = detect_structural_anchor_indices(contour, original_points)
    protected_indices = hard_corner_indices | gap_indices | structural_indices

    working_points = presmooth_contour_points(
        contour,
        original_points,
        protected_indices=protected_indices,
    )

    anchor_points = remove_collinear_points(
        working_points,
        tolerance=max(0.05, config.simplify_tolerance * 0.12),
        protected_indices=protected_indices,
    )
    if not validate_contour_points(contour, anchor_points, config):
        anchor_points = working_points

    smoothed = smooth_between_corners(
        anchor_points,
        corner_indices=protected_indices,
        passes=config.smooth_iterations,
        strength=cleanup_smoothing_strength(contour, config),
    )
    smoothed = collapse_short_edges(
        smoothed,
        min_length=max(0.4, config.merge_distance * 0.9),
        protected_indices=protected_indices,
    )
    smoothed = simplify_closed_preserving_indices(
        smoothed,
        tolerance=cleanup_simplify_tolerance(contour, config),
        protected_indices=protected_indices,
    )
    smoothed = remove_collinear_points(
        smoothed,
        tolerance=max(0.06, config.simplify_tolerance * 0.18),
        protected_indices=protected_indices,
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


def detect_structural_anchor_indices(contour: ClosedContour, points: list[Point]) -> set[int]:
    anchors = _extrema_anchor_indices(points)
    if _is_shadow_like_contour(contour):
        anchors |= _shadow_bridge_anchor_indices(points, contour)
    return anchors


def presmooth_contour_points(
    contour: ClosedContour,
    points: list[Point],
    protected_indices: set[int],
) -> list[Point]:
    if len(points) < 8:
        return points

    if _is_shadow_like_contour(contour):
        return _blend_gaussian_smoothed_points(
            points,
            sigma=1.15,
            blend=0.4,
            protected_indices=protected_indices,
        )

    return points


def cleanup_simplify_tolerance(contour: ClosedContour, config: HybridVectorizerConfig) -> float:
    base_tolerance = min(0.8, max(0.2, config.simplify_tolerance))
    if _is_shadow_like_contour(contour):
        return min(0.85, max(0.45, base_tolerance * 0.95))
    return base_tolerance


def cleanup_smoothing_strength(contour: ClosedContour, config: HybridVectorizerConfig) -> float:
    base_strength = max(0.0, min(0.85, config.smooth_strength))
    if _is_shadow_like_contour(contour):
        return min(base_strength, 0.26)
    return base_strength


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


def detect_hard_corner_indices(points: list[Point], angle_threshold_deg: float = HARD_CORNER_ANGLE_THRESHOLD_DEGREES) -> set[int]:
    if len(points) < 3:
        return set()

    corners = set()
    for index, point in enumerate(points):
        prev_point = points[index - 1]
        next_point = points[(index + 1) % len(points)]
        interior_angle = _corner_angle_degrees(prev_point, point, next_point)
        turn_angle = max(0.0, 180.0 - interior_angle)
        if turn_angle >= angle_threshold_deg:
            corners.add(index)
    return corners


def smooth_between_corners(
    points: list[Point],
    corner_indices: set[int],
    passes: int = 1,
    strength: float = 0.35,
) -> list[Point]:
    if len(points) < 5 or passes <= 0 or strength <= 0.0:
        return points

    protected = {index % len(points) for index in corner_indices}
    if not protected:
        return smooth_closed_contour(points, corner_indices=set(), iterations=passes, strength=strength)

    if _curvature_variance(points) < 0.3:
        passes = max(passes, 3)

    current = points[:]
    strength = max(0.0, min(0.85, strength))
    for _ in range(passes):
        current = _smooth_runs_between_corners(current, protected, strength)
        current = _smooth_runs_between_corners(current, protected, -0.45 * strength)
    return current


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


def _blend_gaussian_smoothed_points(
    points: list[Point],
    sigma: float,
    blend: float,
    protected_indices: set[int],
) -> list[Point]:
    smoothed = gaussian_smooth_closed_contour(points, sigma=sigma)
    protected = {index % len(points) for index in protected_indices}
    blend = max(0.0, min(1.0, blend))

    blended: list[Point] = []
    for index, point in enumerate(points):
        if index in protected:
            blended.append(point)
            continue
        target = smoothed[index]
        blended.append(
            Point(
                x=(1.0 - blend) * point.x + blend * target.x,
                y=(1.0 - blend) * point.y + blend * target.y,
            )
        )
    return blended


def _stabilize_shadow_lower_mass(
    points: list[Point],
    contour: ClosedContour,
    protected_indices: set[int],
) -> list[Point]:
    if len(points) < 8:
        return points

    bbox = contour.bbox
    split_y = bbox.min_y + bbox.height * 0.34
    smoothed = gaussian_smooth_closed_contour(points, sigma=1.3)
    protected = {index % len(points) for index in protected_indices}
    stabilized: list[Point] = []

    for index, point in enumerate(points):
        if index in protected or point.y <= split_y:
            stabilized.append(point)
            continue
        target = smoothed[index]
        stabilized.append(
            Point(
                x=point.x + 0.62 * (target.x - point.x),
                y=point.y + 0.62 * (target.y - point.y),
            )
        )
    return stabilized


def _extrema_anchor_indices(points: list[Point]) -> set[int]:
    if not points:
        return set()

    center_x = sum(point.x for point in points) / len(points)
    center_y = sum(point.y for point in points) / len(points)
    top_index = min(range(len(points)), key=lambda idx: (points[idx].y, abs(points[idx].x - center_x)))
    bottom_index = max(range(len(points)), key=lambda idx: (points[idx].y, -abs(points[idx].x - center_x)))
    left_index = min(range(len(points)), key=lambda idx: (points[idx].x, abs(points[idx].y - center_y)))
    right_index = max(range(len(points)), key=lambda idx: (points[idx].x, -abs(points[idx].y - center_y)))
    return {top_index, bottom_index, left_index, right_index}


def _pin_body_anchor_indices(points: list[Point], contour: ClosedContour) -> set[int]:
    bbox = contour.bbox
    if bbox.width <= 0.0 or bbox.height <= 0.0:
        return set()

    center_x = bbox.min_x + bbox.width * 0.5
    target_y = bbox.min_y + bbox.height * 0.6
    left_candidates = [index for index, point in enumerate(points) if point.x <= center_x]
    right_candidates = [index for index, point in enumerate(points) if point.x >= center_x]
    anchors: set[int] = set()
    if left_candidates:
        anchors.add(
            min(
                left_candidates,
                key=lambda idx: (
                    abs(points[idx].y - target_y) * 1.6,
                    abs(points[idx].x - (bbox.min_x + bbox.width * 0.23)),
                ),
            )
        )
    if right_candidates:
        anchors.add(
            min(
                right_candidates,
                key=lambda idx: (
                    abs(points[idx].y - target_y) * 1.6,
                    abs(points[idx].x - (bbox.max_x - bbox.width * 0.23)),
                ),
            )
        )
    return anchors


def _shadow_bridge_anchor_indices(points: list[Point], contour: ClosedContour) -> set[int]:
    bbox = contour.bbox
    if bbox.width <= 0.0 or bbox.height <= 0.0:
        return set()

    center_x = bbox.min_x + bbox.width * 0.5
    top_rim_limit = bbox.min_y + bbox.height * 0.18
    notch_depth_limit = bbox.min_y + bbox.height * 0.75

    left_bridge_candidates = [
        index
        for index, point in enumerate(points)
        if point.x < center_x and point.y <= top_rim_limit
    ]
    right_bridge_candidates = [
        index
        for index, point in enumerate(points)
        if point.x > center_x and point.y <= top_rim_limit
    ]
    notch_candidates = [
        index
        for index, point in enumerate(points)
        if abs(point.x - center_x) <= bbox.width * 0.18 and point.y <= notch_depth_limit
    ]

    anchors: set[int] = set()
    if left_bridge_candidates:
        anchors.add(max(left_bridge_candidates, key=lambda idx: points[idx].x))
    if right_bridge_candidates:
        anchors.add(min(right_bridge_candidates, key=lambda idx: points[idx].x))
    if notch_candidates:
        anchors.add(
            max(
                notch_candidates,
                key=lambda idx: (points[idx].y, -abs(points[idx].x - center_x)),
            )
        )
    anchors.add(max(range(len(points)), key=lambda idx: points[idx].y))
    return anchors


def _is_pin_body_like_contour(contour: ClosedContour) -> bool:
    if contour.is_hole:
        return False

    bbox = contour.bbox
    if bbox.width <= 0.0 or bbox.height <= 0.0:
        return False

    aspect_ratio = bbox.height / bbox.width
    fill_ratio = contour.area / max(1e-6, bbox.width * bbox.height)
    return aspect_ratio >= 1.18 and 0.46 <= fill_ratio <= 0.9


def _is_shadow_like_contour(contour: ClosedContour) -> bool:
    if contour.is_hole:
        return False

    bbox = contour.bbox
    if bbox.width <= 0.0 or bbox.height <= 0.0:
        return False

    aspect_ratio = bbox.width / bbox.height
    fill_ratio = contour.area / max(1e-6, bbox.width * bbox.height)
    return aspect_ratio >= 1.65 and 0.38 <= fill_ratio <= 0.92


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


def _smooth_runs_between_corners(points: list[Point], protected_indices: set[int], strength: float) -> list[Point]:
    if len(points) < 5 or not protected_indices:
        return points

    updated = points[:]
    for span_indices in _iter_closed_spans_between_corners(len(points), protected_indices):
        if len(span_indices) < 3:
            continue
        span_points = [points[index] for index in span_indices]
        smoothed_span = _smooth_open_span_points(span_points, strength)
        for local_index, global_index in enumerate(span_indices[1:-1], start=1):
            updated[global_index] = smoothed_span[local_index]
    return updated


def _iter_closed_spans_between_corners(count: int, protected_indices: set[int]) -> list[list[int]]:
    if count <= 0 or not protected_indices:
        return [list(range(count))]

    ordered = sorted(index % count for index in protected_indices)
    if len(ordered) == 1:
        start_index = ordered[0]
        return [[(start_index + offset) % count for offset in range(count + 1)]]

    spans: list[list[int]] = []
    for offset, start_index in enumerate(ordered):
        end_index = ordered[(offset + 1) % len(ordered)]
        indices = [start_index]
        cursor = start_index
        while cursor != end_index:
            cursor = (cursor + 1) % count
            indices.append(cursor)
            if len(indices) > count + 1:
                break
        spans.append(indices)
    return spans


def _smooth_open_span_points(points: list[Point], strength: float) -> list[Point]:
    if len(points) < 3:
        return points

    updated = points[:]
    last_index = len(points) - 1
    for index in range(1, last_index):
        point = points[index]
        prev2 = points[max(0, index - 2)]
        prev1 = points[index - 1]
        next1 = points[index + 1]
        next2 = points[min(last_index, index + 2)]
        target = Point(
            x=(prev2.x + 4.0 * prev1.x + 6.0 * point.x + 4.0 * next1.x + next2.x) / 16.0,
            y=(prev2.y + 4.0 * prev1.y + 6.0 * point.y + 4.0 * next1.y + next2.y) / 16.0,
        )
        angle = _corner_angle_degrees(prev1, point, next1)
        curve_factor = _smooth_curve_factor(angle)
        updated[index] = Point(
            x=point.x + strength * curve_factor * (target.x - point.x),
            y=point.y + strength * curve_factor * (target.y - point.y),
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
