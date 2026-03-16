from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, degrees, pi, radians, sin, sqrt

import cv2
import numpy as np

from yd_vector.hybrid_vectorizer.geometry import (
    Point,
    PrimitiveCircle,
    PrimitiveEllipse,
    PrimitiveRectangle,
    PrimitiveRoundedRectangle,
    SegmentArcCircular,
    SegmentArcElliptical,
    distance,
    polygon_area,
    polygon_perimeter,
)


@dataclass(frozen=True)
class CircleFitResult:
    primitive: PrimitiveCircle
    radial_error: float
    max_radial_error: float
    circularity: float
    area_ratio: float
    confidence: float


@dataclass(frozen=True)
class EllipseFitResult:
    primitive: PrimitiveEllipse
    normalized_error: float
    max_normalized_error: float
    area_ratio: float
    confidence: float


@dataclass(frozen=True)
class RectangleFitResult:
    primitive: PrimitiveRectangle
    mean_error: float
    max_error: float
    area_ratio: float
    confidence: float


@dataclass(frozen=True)
class RoundedRectangleFitResult:
    primitive: PrimitiveRoundedRectangle
    mean_error: float
    max_error: float
    area_ratio: float
    confidence: float


@dataclass(frozen=True)
class CircularArcFitResult:
    segment: SegmentArcCircular
    mean_error: float
    max_error: float
    normalized_max_error: float
    sweep_degrees: float
    confidence: float


@dataclass(frozen=True)
class EllipticalArcFitResult:
    segment: SegmentArcElliptical
    mean_error: float
    max_error: float
    normalized_max_error: float
    sweep_degrees: float
    confidence: float


@dataclass(frozen=True)
class _LocalFrame:
    center: Point
    rotation_degrees: float
    local_points: np.ndarray
    half_width: float
    half_height: float


def fit_circle(points: list[Point]) -> CircleFitResult | None:
    if len(points) < 8:
        return None

    coords = np.asarray([(point.x, point.y) for point in points], dtype=np.float32).reshape((-1, 1, 2))
    (cx, cy), radius = cv2.minEnclosingCircle(coords)
    if radius <= 0.0:
        return None

    center = Point(float(cx), float(cy))
    radii = np.sqrt(
        (coords[:, 0, 0].astype(np.float64) - center.x) ** 2
        + (coords[:, 0, 1].astype(np.float64) - center.y) ** 2
    )
    radial_error = float(np.std(radii) / max(1e-6, radius))
    max_radial_error = float(np.max(np.abs(radii - radius)) / max(1e-6, radius))

    area = abs(polygon_area(points))
    perimeter = polygon_perimeter(points)
    circularity = float((4.0 * pi * area) / max(1e-6, perimeter * perimeter))
    expected_area = pi * radius * radius
    area_ratio = float(area / max(1e-6, expected_area))

    confidence = _combine_confidence(
        _scaled_score(radial_error, 0.0, 0.09, invert=True),
        _scaled_score(max_radial_error, 0.0, 0.14, invert=True),
        _scaled_score(circularity, 0.78, 0.995, invert=False),
        _scaled_score(abs(1.0 - area_ratio), 0.0, 0.14, invert=True),
    )
    if circularity > 0.88:
        confidence = max(confidence, 0.96)
    return CircleFitResult(
        primitive=PrimitiveCircle(center=center, radius=radius),
        radial_error=radial_error,
        max_radial_error=max_radial_error,
        circularity=circularity,
        area_ratio=area_ratio,
        confidence=confidence,
    )


def fit_circular_arc(points: list[Point]) -> CircularArcFitResult | None:
    if len(points) < 4:
        return None

    fit = _fit_circle_parameters(points)
    if fit is None:
        return None

    center, radius, radii = fit
    coords = np.asarray([(point.x, point.y) for point in points], dtype=np.float64)
    angles = np.unwrap(np.arctan2(coords[:, 1] - center.y, coords[:, 0] - center.x))
    sweep_radians = float(angles[-1] - angles[0])
    sweep_degrees = abs(degrees(sweep_radians))
    if sweep_degrees <= 1.0:
        return None

    projected = np.column_stack([center.x + radius * np.cos(angles), center.y + radius * np.sin(angles)])
    errors = np.linalg.norm(coords - projected, axis=1)
    mean_error = float(np.mean(errors))
    max_error = float(np.max(errors))
    normalized_max_error = float(max_error / max(1.0, radius))

    chord = distance(points[0], points[-1])
    if chord > (2.0 * radius + 1e-3):
        return None

    segment = SegmentArcCircular(
        start=points[0],
        end=points[-1],
        radius=radius,
        large_arc=abs(sweep_radians) > pi,
        sweep=sweep_radians > 0.0,
    )
    confidence = _combine_confidence(
        _scaled_score(normalized_max_error, 0.0, 0.12, invert=True),
        _scaled_score(float(np.std(radii) / max(1e-6, radius)), 0.0, 0.10, invert=True),
        _scaled_score(sweep_degrees, 18.0, 170.0, invert=False),
    )
    return CircularArcFitResult(
        segment=segment,
        mean_error=mean_error,
        max_error=max_error,
        normalized_max_error=normalized_max_error,
        sweep_degrees=sweep_degrees,
        confidence=confidence,
    )


def fit_rotated_ellipse(points: list[Point]) -> EllipseFitResult | None:
    if len(points) < 8:
        return None

    frame = _fit_local_frame(points)
    if frame is None:
        return None

    rotated = frame.local_points
    radius_x = frame.half_width
    radius_y = frame.half_height
    if radius_x <= 0.0 or radius_y <= 0.0:
        return None

    normalized = (rotated[:, 0] / radius_x) ** 2 + (rotated[:, 1] / radius_y) ** 2
    normalized_error = float(np.mean(np.abs(normalized - 1.0)))
    max_normalized_error = float(np.max(np.abs(normalized - 1.0)))

    area = abs(polygon_area(points))
    expected_area = pi * radius_x * radius_y
    area_ratio = float(area / max(1e-6, expected_area))

    confidence = _combine_confidence(
        _scaled_score(normalized_error, 0.0, 0.14, invert=True),
        _scaled_score(max_normalized_error, 0.0, 0.22, invert=True),
        _scaled_score(abs(1.0 - area_ratio), 0.0, 0.18, invert=True),
    )

    primitive = PrimitiveEllipse(
        center=frame.center,
        radius_x=radius_x,
        radius_y=radius_y,
        rotation_degrees=frame.rotation_degrees,
    )
    return EllipseFitResult(
        primitive=primitive,
        normalized_error=normalized_error,
        max_normalized_error=max_normalized_error,
        area_ratio=area_ratio,
        confidence=confidence,
    )


def fit_elliptical_arc(points: list[Point]) -> EllipticalArcFitResult | None:
    if len(points) < 5:
        return None

    result = fit_rotated_ellipse(points)
    if result is None:
        return None

    primitive = result.primitive
    if primitive.radius_x <= 0.0 or primitive.radius_y <= 0.0:
        return None

    local = _to_ellipse_local(points, primitive)
    angles = np.unwrap(np.arctan2(local[:, 1] / primitive.radius_y, local[:, 0] / primitive.radius_x))
    sweep_radians = float(angles[-1] - angles[0])
    sweep_degrees = abs(degrees(sweep_radians))
    if sweep_degrees <= 1.0:
        return None

    projected_local = np.column_stack(
        [
            primitive.radius_x * np.cos(angles),
            primitive.radius_y * np.sin(angles),
        ]
    )
    errors = np.linalg.norm(local - projected_local, axis=1)
    mean_error = float(np.mean(errors))
    max_error = float(np.max(errors))
    normalized_max_error = float(max_error / max(1.0, min(primitive.radius_x, primitive.radius_y)))

    segment = SegmentArcElliptical(
        start=points[0],
        end=points[-1],
        radius_x=primitive.radius_x,
        radius_y=primitive.radius_y,
        rotation_degrees=primitive.rotation_degrees,
        large_arc=abs(sweep_radians) > pi,
        sweep=sweep_radians > 0.0,
    )
    confidence = _combine_confidence(
        _scaled_score(normalized_max_error, 0.0, 0.16, invert=True),
        _scaled_score(result.normalized_error, 0.0, 0.20, invert=True),
        _scaled_score(sweep_degrees, 18.0, 170.0, invert=False),
        result.confidence,
    )
    return EllipticalArcFitResult(
        segment=segment,
        mean_error=mean_error,
        max_error=max_error,
        normalized_max_error=normalized_max_error,
        sweep_degrees=sweep_degrees,
        confidence=confidence,
    )


def fit_rectangle(points: list[Point]) -> RectangleFitResult | None:
    if len(points) < 4:
        return None

    frame = _fit_local_frame(points)
    if frame is None or frame.half_width <= 0.0 or frame.half_height <= 0.0:
        return None

    errors = np.asarray(
        [
            _sd_box(float(local[0]), float(local[1]), frame.half_width, frame.half_height)
            for local in frame.local_points
        ],
        dtype=np.float64,
    )
    mean_error = float(np.mean(np.abs(errors)))
    max_error = float(np.max(np.abs(errors)))

    primitive = PrimitiveRectangle(
        center=frame.center,
        width=frame.half_width * 2.0,
        height=frame.half_height * 2.0,
        rotation_degrees=frame.rotation_degrees,
    )
    area = abs(polygon_area(points))
    expected_area = primitive.width * primitive.height
    area_ratio = float(area / max(1e-6, expected_area))
    norm_scale = max(1.0, min(frame.half_width, frame.half_height))
    confidence = _combine_confidence(
        _scaled_score(mean_error / norm_scale, 0.0, 0.06, invert=True),
        _scaled_score(max_error / norm_scale, 0.0, 0.12, invert=True),
        _scaled_score(abs(1.0 - area_ratio), 0.0, 0.14, invert=True),
    )
    return RectangleFitResult(
        primitive=primitive,
        mean_error=mean_error,
        max_error=max_error,
        area_ratio=area_ratio,
        confidence=confidence,
    )


def fit_rounded_rectangle(points: list[Point]) -> RoundedRectangleFitResult | None:
    if len(points) < 8:
        return None

    frame = _fit_local_frame(points)
    if frame is None or frame.half_width <= 0.0 or frame.half_height <= 0.0:
        return None

    max_radius = max(0.0, min(frame.half_width, frame.half_height) * 0.48)
    if max_radius <= 1.0:
        return None

    radius_candidates = np.linspace(max_radius * 0.10, max_radius, 16, dtype=np.float64)
    best_radius = None
    best_mean_error = float("inf")
    best_max_error = float("inf")
    for radius in radius_candidates:
        errors = np.asarray(
            [
                _sd_round_box(float(local[0]), float(local[1]), frame.half_width, frame.half_height, float(radius))
                for local in frame.local_points
            ],
            dtype=np.float64,
        )
        mean_error = float(np.mean(np.abs(errors)))
        if mean_error < best_mean_error:
            best_mean_error = mean_error
            best_max_error = float(np.max(np.abs(errors)))
            best_radius = float(radius)

    if best_radius is None or best_radius <= 0.0:
        return None

    primitive = PrimitiveRoundedRectangle(
        center=frame.center,
        width=frame.half_width * 2.0,
        height=frame.half_height * 2.0,
        corner_radius=best_radius,
        rotation_degrees=frame.rotation_degrees,
    )
    area = abs(polygon_area(points))
    expected_area = primitive.width * primitive.height - (4.0 - pi) * best_radius * best_radius
    area_ratio = float(area / max(1e-6, expected_area))
    norm_scale = max(1.0, min(frame.half_width, frame.half_height))
    radius_ratio = best_radius / max(1e-6, min(frame.half_width, frame.half_height))
    confidence = _combine_confidence(
        _scaled_score(best_mean_error / norm_scale, 0.0, 0.07, invert=True),
        _scaled_score(best_max_error / norm_scale, 0.0, 0.14, invert=True),
        _scaled_score(abs(1.0 - area_ratio), 0.0, 0.18, invert=True),
        _scaled_score(radius_ratio, 0.12, 0.38, invert=False),
    )
    return RoundedRectangleFitResult(
        primitive=primitive,
        mean_error=best_mean_error,
        max_error=best_max_error,
        area_ratio=area_ratio,
        confidence=confidence,
    )


def detect_narrow_gap_indices(
    points: list[Point],
    gap_distance: float,
    protect_span: int = 2,
    min_index_separation: int = 3,
) -> set[int]:
    if len(points) < 8 or gap_distance <= 0.0:
        return set()

    count = len(points)
    protected: set[int] = set()
    for left in range(count):
        for right in range(left + min_index_separation, count):
            arc_distance = min(right - left, count - (right - left))
            if arc_distance < min_index_separation:
                continue
            if distance(points[left], points[right]) > gap_distance:
                continue
            for offset in range(-protect_span, protect_span + 1):
                protected.add((left + offset) % count)
                protected.add((right + offset) % count)
    return protected


def _fit_circle_parameters(points: list[Point]) -> tuple[Point, float, np.ndarray] | None:
    coords = np.asarray([(point.x, point.y) for point in points], dtype=np.float64)
    xs = coords[:, 0]
    ys = coords[:, 1]
    matrix = np.column_stack([xs, ys, np.ones(len(points), dtype=np.float64)])
    rhs = -(xs * xs + ys * ys)

    try:
        a, b, c = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None

    center = Point(float(-0.5 * a), float(-0.5 * b))
    radius_sq = center.x * center.x + center.y * center.y - c
    if radius_sq <= 0.0:
        return None

    radius = sqrt(radius_sq)
    radii = np.sqrt((xs - center.x) ** 2 + (ys - center.y) ** 2)
    return center, radius, radii


def _to_ellipse_local(points: list[Point], ellipse: PrimitiveEllipse) -> np.ndarray:
    angle = radians(ellipse.rotation_degrees)
    cos_a = cos(angle)
    sin_a = sin(angle)
    coords = np.asarray([(point.x - ellipse.center.x, point.y - ellipse.center.y) for point in points], dtype=np.float64)
    x_local = coords[:, 0] * cos_a + coords[:, 1] * sin_a
    y_local = -coords[:, 0] * sin_a + coords[:, 1] * cos_a
    return np.column_stack([x_local, y_local])


def _fit_local_frame(points: list[Point]) -> _LocalFrame | None:
    coords = np.asarray([(point.x, point.y) for point in points], dtype=np.float64)
    center_xy = coords.mean(axis=0)
    centered = coords - center_xy
    cov = np.cov(centered.T)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None

    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]
    rotation = float(degrees(atan2(eigenvectors[1, 0], eigenvectors[0, 0])))

    rotated = centered @ eigenvectors
    min_x = float(rotated[:, 0].min())
    max_x = float(rotated[:, 0].max())
    min_y = float(rotated[:, 1].min())
    max_y = float(rotated[:, 1].max())
    center_local = np.asarray([(min_x + max_x) * 0.5, (min_y + max_y) * 0.5], dtype=np.float64)
    local_points = rotated - center_local
    center_world = center_xy + center_local @ eigenvectors.T
    return _LocalFrame(
        center=Point(float(center_world[0]), float(center_world[1])),
        rotation_degrees=rotation,
        local_points=local_points,
        half_width=(max_x - min_x) * 0.5,
        half_height=(max_y - min_y) * 0.5,
    )


def _sd_box(x: float, y: float, half_width: float, half_height: float) -> float:
    dx = abs(x) - half_width
    dy = abs(y) - half_height
    outside_x = max(dx, 0.0)
    outside_y = max(dy, 0.0)
    outside = sqrt(outside_x * outside_x + outside_y * outside_y)
    inside = min(max(dx, dy), 0.0)
    return outside + inside


def _sd_round_box(x: float, y: float, half_width: float, half_height: float, radius: float) -> float:
    radius = min(radius, half_width, half_height)
    qx = abs(x) - (half_width - radius)
    qy = abs(y) - (half_height - radius)
    outside_x = max(qx, 0.0)
    outside_y = max(qy, 0.0)
    outside = sqrt(outside_x * outside_x + outside_y * outside_y)
    inside = min(max(qx, qy), 0.0)
    return outside + inside - radius


def _scaled_score(value: float, low: float, high: float, invert: bool) -> float:
    if high <= low:
        return 1.0
    normalized = (value - low) / (high - low)
    normalized = max(0.0, min(1.0, normalized))
    return 1.0 - normalized if invert else normalized


def _combine_confidence(*scores: float) -> float:
    if not scores:
        return 0.0
    return float(sum(max(0.0, min(1.0, score)) for score in scores) / len(scores))
