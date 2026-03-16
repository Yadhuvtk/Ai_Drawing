from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, degrees, pi, radians, sin, sqrt

import cv2
import numpy as np

from yd_vector.hybrid_vectorizer.geometry import (
    ClosedContour,
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


@dataclass(frozen=True)
class ContourDescriptors:
    area: float
    perimeter: float
    centroid: Point
    circularity: float
    convex_hull: np.ndarray
    solidity: float
    radial_angles: np.ndarray
    radial_distances: np.ndarray
    principal_orientation_degrees: float
    corner_count: int
    curvature_extrema_count: int
    bbox_aspect_ratio: float
    local_frame: _LocalFrame | None


@dataclass(frozen=True)
class ShapeClassification:
    representation: str
    kind: str
    confidence: float
    params: dict[str, object]
    descriptors: dict[str, object]
    segments: tuple[object, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "representation": self.representation,
            "kind": self.kind,
            "confidence": self.confidence,
            "params": self.params,
            "descriptors": self.descriptors,
            "segments": list(self.segments),
        }


PARAMETERIZED_CONFIDENCE_THRESHOLDS: dict[str, float] = {
    "circle": 0.90,
    "ellipse": 0.88,
    "rounded_rectangle": 0.84,
    "triangle": 0.84,
    "star": 0.86,
    "capsule": 0.82,
    "dshape": 0.82,
}

RADIAL_SIGNATURE_SAMPLES = 144


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


def compute_contour_descriptors(points: list[Point]) -> ContourDescriptors | None:
    if len(points) < 3:
        return None

    coords = np.asarray([(point.x, point.y) for point in points], dtype=np.float64)
    area = abs(polygon_area(points))
    perimeter = polygon_perimeter(points)
    centroid = _polygon_centroid(points)
    circularity = float((4.0 * pi * area) / max(1e-6, perimeter * perimeter))

    hull = cv2.convexHull(coords.astype(np.float32).reshape((-1, 1, 2)))
    hull_area = float(abs(cv2.contourArea(hull)))
    solidity = float(area / max(1e-6, hull_area))

    radial_angles, radial_distances = _sample_radial_signature(coords, centroid, sample_count=RADIAL_SIGNATURE_SAMPLES)
    frame = _fit_local_frame(points)
    turning_angles = _turning_angle_signal(points)
    smoothed_turning = _smooth_circular_signal(turning_angles, radius=2)
    corner_count = int(np.sum(smoothed_turning >= 35.0))
    curvature_extrema_count = _count_local_extrema(smoothed_turning, minimum_value=18.0)

    bbox_width = float(coords[:, 0].max() - coords[:, 0].min())
    bbox_height = float(coords[:, 1].max() - coords[:, 1].min())
    bbox_aspect_ratio = bbox_width / max(1e-6, bbox_height)
    principal_orientation = frame.rotation_degrees if frame is not None else 0.0

    return ContourDescriptors(
        area=area,
        perimeter=perimeter,
        centroid=centroid,
        circularity=circularity,
        convex_hull=hull.reshape(-1, 2),
        solidity=solidity,
        radial_angles=radial_angles,
        radial_distances=radial_distances,
        principal_orientation_degrees=principal_orientation,
        corner_count=corner_count,
        curvature_extrema_count=curvature_extrema_count,
        bbox_aspect_ratio=bbox_aspect_ratio,
        local_frame=frame,
    )


def analyze_loop(loop_or_points: ClosedContour | list[Point]) -> dict[str, object]:
    points = _coerce_loop_points(loop_or_points)
    descriptors = compute_contour_descriptors(points)
    if descriptors is None:
        return {
            "area": 0.0,
            "perimeter": 0.0,
            "centroid": {"x": 0.0, "y": 0.0},
            "circularity": 0.0,
            "solidity": 0.0,
            "convex_hull_points": [],
            "principal_orientation_degrees": 0.0,
            "radial_signature": [],
            "corner_count": 0,
            "curvature_extrema_count": 0,
            "bbox_aspect_ratio": 0.0,
            "symmetry_hint": 0.0,
        }
    return _descriptor_dict(descriptors)


def detect_circle_candidate(
    points: list[Point],
    descriptors: ContourDescriptors | None = None,
) -> dict[str, object] | None:
    descriptors = descriptors or compute_contour_descriptors(points)
    if descriptors is None:
        return None

    result = fit_circle(points)
    if result is None:
        return None

    blended_confidence = _combine_confidence(
        result.confidence,
        _scaled_score(abs(descriptors.bbox_aspect_ratio - 1.0), 0.0, 0.2, invert=True),
        _scaled_score(descriptors.corner_count, 0.0, 6.0, invert=True),
        _scaled_score(abs(1.0 - descriptors.solidity), 0.0, 0.08, invert=True),
    )
    confidence = max(result.confidence, blended_confidence)
    return {
        "representation": "parameterized",
        "kind": "circle",
        "confidence": confidence,
        "params": {
            "cx": result.primitive.center.x,
            "cy": result.primitive.center.y,
            "r": result.primitive.radius,
            "primitive": result.primitive,
            "fit_result": result,
        },
        "segments": [],
    }


def detect_ellipse_candidate(
    points: list[Point],
    descriptors: ContourDescriptors | None = None,
) -> dict[str, object] | None:
    descriptors = descriptors or compute_contour_descriptors(points)
    if descriptors is None:
        return None

    ellipse_result = fit_rotated_ellipse(points)
    if ellipse_result is None:
        return None

    circle_candidate = detect_circle_candidate(points, descriptors=descriptors)
    primitive = ellipse_result.primitive
    anisotropy = abs(primitive.radius_x - primitive.radius_y) / max(1e-6, max(primitive.radius_x, primitive.radius_y))
    if circle_candidate is not None and anisotropy <= 0.08 and float(circle_candidate["confidence"]) >= ellipse_result.confidence:
        return None

    blended_confidence = _combine_confidence(
        ellipse_result.confidence,
        _scaled_score(anisotropy, 0.06, 0.45, invert=False),
        _scaled_score(descriptors.corner_count, 0.0, 10.0, invert=True),
        _scaled_score(abs(1.0 - descriptors.solidity), 0.0, 0.16, invert=True),
    )
    confidence = max(ellipse_result.confidence * 0.96, blended_confidence)
    return {
        "representation": "parameterized",
        "kind": "ellipse",
        "confidence": confidence,
        "params": {
            "cx": primitive.center.x,
            "cy": primitive.center.y,
            "rx": primitive.radius_x,
            "ry": primitive.radius_y,
            "theta": primitive.rotation_degrees,
            "primitive": primitive,
            "fit_result": ellipse_result,
        },
        "segments": [],
    }


def detect_rounded_rectangle_candidate(
    points: list[Point],
    descriptors: ContourDescriptors | None = None,
) -> dict[str, object] | None:
    descriptors = descriptors or compute_contour_descriptors(points)
    if descriptors is None:
        return None

    rounded_result = fit_rounded_rectangle(points)
    if rounded_result is not None:
        primitive = rounded_result.primitive
        blended_confidence = _combine_confidence(
            rounded_result.confidence,
            _scaled_score(descriptors.solidity, 0.82, 1.0, invert=False),
            _scaled_score(abs(descriptors.corner_count - 4.0), 0.0, 6.0, invert=True),
        )
        confidence = max(rounded_result.confidence, blended_confidence)
        return {
            "representation": "parameterized",
            "kind": "rounded_rectangle",
            "confidence": confidence,
            "params": {
                "cx": primitive.center.x,
                "cy": primitive.center.y,
                "w": primitive.width,
                "h": primitive.height,
                "theta": primitive.rotation_degrees,
                "corner_radius": primitive.corner_radius,
                "primitive": primitive,
                "fit_result": rounded_result,
            },
            "segments": [],
        }

    rectangle_result = fit_rectangle(points)
    if rectangle_result is None:
        return None

    primitive = rectangle_result.primitive
    rounded_primitive = PrimitiveRoundedRectangle(
        center=primitive.center,
        width=primitive.width,
        height=primitive.height,
        corner_radius=0.0,
        rotation_degrees=primitive.rotation_degrees,
    )
    blended_confidence = _combine_confidence(
        rectangle_result.confidence,
        _scaled_score(descriptors.solidity, 0.86, 1.0, invert=False),
        _scaled_score(abs(descriptors.corner_count - 4.0), 0.0, 4.0, invert=True),
    )
    confidence = max(rectangle_result.confidence * 0.9, blended_confidence)
    return {
        "representation": "parameterized",
        "kind": "rounded_rectangle",
        "confidence": confidence * 0.92,
        "params": {
            "cx": rounded_primitive.center.x,
            "cy": rounded_primitive.center.y,
            "w": rounded_primitive.width,
            "h": rounded_primitive.height,
            "theta": rounded_primitive.rotation_degrees,
            "corner_radius": rounded_primitive.corner_radius,
            "primitive": rounded_primitive,
            "fit_result": rectangle_result,
        },
        "segments": [],
    }


def detect_isosceles_triangle_candidate(
    points: list[Point],
    descriptors: ContourDescriptors | None = None,
) -> dict[str, object] | None:
    descriptors = descriptors or compute_contour_descriptors(points)
    if descriptors is None:
        return None

    approx = _approx_hull_polygon(descriptors.convex_hull, target_vertices=3)
    if approx is None or len(approx) != 3:
        return None

    vertices = [Point(float(x), float(y)) for x, y in approx]
    side_lengths = [distance(vertices[index], vertices[(index + 1) % 3]) for index in range(3)]
    equal_pair, equal_score = _best_equal_side_pair(side_lengths)
    if equal_pair is None or equal_score < 0.78:
        return None

    apex_index = {frozenset((0, 1)): 1, frozenset((1, 2)): 2, frozenset((0, 2)): 0}[frozenset(equal_pair)]
    base_indices = [index for index in range(3) if index != apex_index]
    base_mid = Point(
        (vertices[base_indices[0]].x + vertices[base_indices[1]].x) * 0.5,
        (vertices[base_indices[0]].y + vertices[base_indices[1]].y) * 0.5,
    )
    theta = float(degrees(atan2(base_mid.y - vertices[apex_index].y, base_mid.x - vertices[apex_index].x)))

    triangle_area = abs(polygon_area(vertices))
    area_ratio = descriptors.area / max(1e-6, triangle_area)
    confidence = _combine_confidence(
        equal_score,
        _scaled_score(abs(1.0 - area_ratio), 0.0, 0.22, invert=True),
        _scaled_score(descriptors.solidity, 0.82, 1.0, invert=False),
        _scaled_score(abs(descriptors.corner_count - 3.0), 0.0, 4.0, invert=True),
    )
    return {
        "representation": "parameterized",
        "kind": "triangle",
        "confidence": confidence,
        "params": {
            "cx": descriptors.centroid.x,
            "cy": descriptors.centroid.y,
            "vertices": vertices,
            "corner_radius": 0.0,  # TODO: estimate rounded triangle corners from curvature spans.
            "theta": theta,
        },
        "segments": [],
    }


def detect_star_candidate(
    points: list[Point],
    min_points: int = 3,
    max_points: int = 8,
    descriptors: ContourDescriptors | None = None,
) -> dict[str, object] | None:
    descriptors = descriptors or compute_contour_descriptors(points)
    if descriptors is None or len(descriptors.radial_distances) < 2 * min_points:
        return None
    if descriptors.solidity > 0.93:
        return None

    best_candidate: dict[str, object] | None = None
    for star_points in range(min_points, max_points + 1):
        star_fit = _evaluate_star_signature(
            descriptors.radial_angles,
            descriptors.radial_distances,
            star_points,
        )
        if star_fit is None:
            continue

        periodicity = float(star_fit["periodicity"])
        contrast = float(star_fit["contrast"])
        confidence = _combine_confidence(
            _scaled_score(contrast, 0.12, 0.42, invert=False),
            _scaled_score(periodicity, 0.45, 0.98, invert=False),
            _scaled_score(1.0 - descriptors.solidity, 0.06, 0.42, invert=False),
            _scaled_score(abs(descriptors.corner_count - (2.0 * star_points)), 0.0, 8.0, invert=True),
        )
        candidate = {
            "representation": "parameterized",
            "kind": "star",
            "confidence": confidence,
            "params": {
                "cx": descriptors.centroid.x,
                "cy": descriptors.centroid.y,
                "n": star_points,
                "r_outer": float(star_fit["r_outer"]),
                "r_inner": float(star_fit["r_inner"]),
                "theta": float(star_fit["theta"]),
                "corner_radius": 0.0,  # TODO: estimate rounded star corners from contour curvature spans.
            },
            "segments": [],
        }
        if best_candidate is None or float(candidate["confidence"]) > float(best_candidate["confidence"]):
            best_candidate = candidate
    return best_candidate


def detect_dshape_candidate(
    points: list[Point],
    descriptors: ContourDescriptors | None = None,
) -> dict[str, object] | None:
    descriptors = descriptors or compute_contour_descriptors(points)
    if descriptors is None or descriptors.local_frame is None:
        return None

    # TODO: Add a stronger D-shape detector once a few stable examples are available.
    frame = descriptors.local_frame
    local = frame.local_points
    min_dimension = max(1e-6, min(frame.half_width, frame.half_height))
    flat_scores = {
        "left": float(np.mean(np.abs(local[:, 0] + frame.half_width) <= 0.08 * min_dimension)),
        "right": float(np.mean(np.abs(local[:, 0] - frame.half_width) <= 0.08 * min_dimension)),
        "top": float(np.mean(np.abs(local[:, 1] + frame.half_height) <= 0.08 * min_dimension)),
        "bottom": float(np.mean(np.abs(local[:, 1] - frame.half_height) <= 0.08 * min_dimension)),
    }
    flat_side, flat_support = max(flat_scores.items(), key=lambda item: item[1])
    confidence = _combine_confidence(
        _scaled_score(flat_support, 0.18, 0.40, invert=False),
        _scaled_score(descriptors.solidity, 0.90, 1.0, invert=False),
    )
    if confidence < 0.68:
        return None
    return {
        "representation": "parameterized",
        "kind": "dshape",
        "confidence": confidence,
        "params": {
            "cx": frame.center.x,
            "cy": frame.center.y,
            "w": frame.half_width * 2.0,
            "h": frame.half_height * 2.0,
            "theta": frame.rotation_degrees,
            "flat_side": flat_side,
        },
        "segments": [],
    }


def classify_parameterized_shape(
    points: list[Point],
    metadata: dict[str, object] | None = None,
) -> dict[str, object] | None:
    descriptors = compute_contour_descriptors(points)
    if descriptors is None:
        return None

    metadata = metadata or {}
    hole_like = bool(metadata.get("is_hole", False))
    candidates = [
        detect_circle_candidate(points, descriptors=descriptors),
        detect_ellipse_candidate(points, descriptors=descriptors),
        detect_rounded_rectangle_candidate(points, descriptors=descriptors),
    ]
    if not hole_like:
        candidates.extend(
            [
                detect_isosceles_triangle_candidate(points, descriptors=descriptors),
                detect_star_candidate(points, descriptors=descriptors),
                detect_dshape_candidate(points, descriptors=descriptors),
            ]
        )

    filtered = [
        candidate
        for candidate in candidates
        if candidate is not None
        and float(candidate["confidence"]) >= PARAMETERIZED_CONFIDENCE_THRESHOLDS.get(str(candidate["kind"]), 1.0)
    ]
    if not filtered:
        return None
    return max(filtered, key=lambda candidate: float(candidate["confidence"]))


def classify_shape_structure(
    loop: ClosedContour | list[Point],
    holes: list[ClosedContour] | None = None,
    metadata: dict[str, object] | None = None,
) -> ShapeClassification:
    points = _coerce_loop_points(loop)
    descriptors = compute_contour_descriptors(points)
    descriptor_payload = _descriptor_dict(descriptors)

    classification_metadata = dict(metadata or {})
    if isinstance(loop, ClosedContour):
        classification_metadata.setdefault("is_hole", bool(loop.is_hole))
        classification_metadata.setdefault("loop_id", loop.contour_id)
    classification_metadata.setdefault("hole_count", len(holes or []))

    candidate = classify_parameterized_shape(points, metadata=classification_metadata)
    if candidate is None:
        return ShapeClassification(
            representation="freeform",
            kind="freeform",
            confidence=1.0,
            params={},
            descriptors=descriptor_payload,
            segments=(),
        )

    return ShapeClassification(
        representation=str(candidate.get("representation", "parameterized")),
        kind=str(candidate.get("kind", "freeform")),
        confidence=float(candidate.get("confidence", 0.0)),
        params=dict(candidate.get("params", {})) if isinstance(candidate.get("params", {}), dict) else {},
        descriptors=descriptor_payload,
        segments=tuple(candidate.get("segments", [])) if isinstance(candidate.get("segments", []), list) else (),
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


def _coerce_loop_points(loop_or_points: ClosedContour | list[Point]) -> list[Point]:
    if isinstance(loop_or_points, ClosedContour):
        return loop_or_points.points
    return list(loop_or_points)


def _descriptor_dict(descriptors: ContourDescriptors | None) -> dict[str, object]:
    if descriptors is None:
        return {}

    radial_signature = [
        {"angle": float(angle), "radius": float(radius)}
        for angle, radius in zip(descriptors.radial_angles.tolist(), descriptors.radial_distances.tolist())
    ]
    convex_hull_points = [
        (float(point[0]), float(point[1]))
        for point in descriptors.convex_hull.tolist()
    ]
    return {
        "area": descriptors.area,
        "perimeter": descriptors.perimeter,
        "centroid": {"x": descriptors.centroid.x, "y": descriptors.centroid.y},
        "circularity": descriptors.circularity,
        "solidity": descriptors.solidity,
        "convex_hull_points": convex_hull_points,
        "principal_orientation_degrees": descriptors.principal_orientation_degrees,
        "radial_signature": radial_signature,
        "corner_count": descriptors.corner_count,
        "curvature_extrema_count": descriptors.curvature_extrema_count,
        "bbox_aspect_ratio": descriptors.bbox_aspect_ratio,
        "symmetry_hint": _radial_symmetry_hint(descriptors.radial_distances),
    }


def _radial_symmetry_hint(radial_distances: np.ndarray) -> float:
    if radial_distances.size < 8:
        return 0.0

    half_turn = radial_distances.size // 2
    rotated = np.roll(radial_distances, half_turn)
    mean_radius = float(np.mean(radial_distances))
    if mean_radius <= 1e-6:
        return 0.0
    mismatch = float(np.mean(np.abs(radial_distances - rotated)) / mean_radius)
    return float(max(0.0, min(1.0, 1.0 - mismatch)))


def _polygon_centroid(points: list[Point]) -> Point:
    signed_area = polygon_area(points)
    if abs(signed_area) <= 1e-8:
        xs = [point.x for point in points]
        ys = [point.y for point in points]
        return Point(float(sum(xs) / len(xs)), float(sum(ys) / len(ys)))

    factor = 1.0 / (6.0 * signed_area)
    accum_x = 0.0
    accum_y = 0.0
    for index, point in enumerate(points):
        nxt = points[(index + 1) % len(points)]
        cross = point.x * nxt.y - nxt.x * point.y
        accum_x += (point.x + nxt.x) * cross
        accum_y += (point.y + nxt.y) * cross
    return Point(float(accum_x * factor), float(accum_y * factor))


def _sample_radial_signature(
    coords: np.ndarray,
    centroid: Point,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    centered = coords - np.asarray([centroid.x, centroid.y], dtype=np.float64)
    angles = np.mod(np.arctan2(centered[:, 1], centered[:, 0]), 2.0 * pi)
    radii = np.linalg.norm(centered, axis=1)
    order = np.argsort(angles)
    angles = angles[order]
    radii = radii[order]

    target_angles = np.linspace(0.0, 2.0 * pi, sample_count, endpoint=False, dtype=np.float64)
    extended_angles = np.concatenate([angles[-1:] - 2.0 * pi, angles, angles[:1] + 2.0 * pi])
    extended_radii = np.concatenate([radii[-1:], radii, radii[:1]])
    sampled = np.interp(target_angles, extended_angles, extended_radii)
    return target_angles, sampled


def _turning_angle_signal(points: list[Point]) -> np.ndarray:
    values = []
    for index, point in enumerate(points):
        prev_point = points[index - 1]
        next_point = points[(index + 1) % len(points)]
        interior_angle = _corner_angle_degrees(prev_point, point, next_point)
        values.append(max(0.0, 180.0 - interior_angle))
    return np.asarray(values, dtype=np.float64)


def _smooth_circular_signal(signal: np.ndarray, radius: int) -> np.ndarray:
    if len(signal) == 0 or radius <= 0:
        return np.asarray(signal, dtype=np.float64)

    current = np.asarray(signal, dtype=np.float64)
    padded = np.pad(current, (radius, radius), mode="wrap")
    kernel = np.ones((2 * radius) + 1, dtype=np.float64) / float((2 * radius) + 1)
    return np.convolve(padded, kernel, mode="valid")


def _count_local_extrema(signal: np.ndarray, minimum_value: float) -> int:
    if len(signal) < 3:
        return 0

    extrema = 0
    for index, value in enumerate(signal):
        prev_value = signal[index - 1]
        next_value = signal[(index + 1) % len(signal)]
        if value >= minimum_value and value >= prev_value and value >= next_value and (value > prev_value or value > next_value):
            extrema += 1
    return extrema


def _approx_hull_polygon(hull_points: np.ndarray, target_vertices: int) -> np.ndarray | None:
    contour = np.asarray(hull_points, dtype=np.float32).reshape((-1, 1, 2))
    if len(contour) < target_vertices:
        return None

    perimeter = float(cv2.arcLength(contour, True))
    best: np.ndarray | None = None
    best_delta = float("inf")
    for fraction in np.linspace(0.01, 0.18, 18, dtype=np.float64):
        approx = cv2.approxPolyDP(contour, float(perimeter * fraction), True).reshape(-1, 2)
        delta = abs(len(approx) - target_vertices)
        if delta < best_delta:
            best = approx
            best_delta = delta
        if len(approx) == target_vertices:
            return approx
    if best is not None and len(best) == target_vertices:
        return best
    return None


def _best_equal_side_pair(side_lengths: list[float]) -> tuple[tuple[int, int] | None, float]:
    best_pair: tuple[int, int] | None = None
    best_score = 0.0
    pairs = ((0, 1), (1, 2), (0, 2))
    for left, right in pairs:
        score = 1.0 - abs(side_lengths[left] - side_lengths[right]) / max(1e-6, max(side_lengths[left], side_lengths[right]))
        if score > best_score:
            best_score = score
            best_pair = (left, right)
    return best_pair, float(best_score)


def _evaluate_star_signature(
    radial_angles: np.ndarray,
    radial_distances: np.ndarray,
    point_count: int,
) -> dict[str, float] | None:
    if len(radial_distances) < 2 * point_count:
        return None

    sector_count = 2 * point_count
    max_shift = max(1, len(radial_distances) // sector_count)
    best: dict[str, float] | None = None
    for shift in range(max_shift):
        rolled = np.roll(radial_distances, -shift)
        sector_values = np.asarray([np.mean(chunk) for chunk in np.array_split(rolled, sector_count)], dtype=np.float64)
        outer = sector_values[0::2]
        inner = sector_values[1::2]
        if len(outer) != point_count or len(inner) != point_count:
            continue

        mean_radius = max(1e-6, float(sector_values.mean()))
        contrast = float((outer.mean() - inner.mean()) / mean_radius)
        periodicity = float(
            1.0
            - 0.5
            * (
                np.std(outer) / max(1e-6, float(outer.mean()))
                + np.std(inner) / max(1e-6, float(inner.mean()))
            )
        )
        score = max(0.0, contrast) * max(0.0, periodicity)
        if best is None or score > best["score"]:
            best = {
                "score": float(score),
                "contrast": contrast,
                "periodicity": periodicity,
                "r_outer": float(outer.mean()),
                "r_inner": float(inner.mean()),
                "theta": float(degrees(radial_angles[shift % len(radial_angles)])),
            }
    if best is None or best["r_outer"] <= best["r_inner"]:
        return None
    return best


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


def _corner_angle_degrees(prev_point: Point, point: Point, next_point: Point) -> float:
    ux = prev_point.x - point.x
    uy = prev_point.y - point.y
    vx = next_point.x - point.x
    vy = next_point.y - point.y

    mag_u = sqrt((ux * ux) + (uy * uy))
    mag_v = sqrt((vx * vx) + (vy * vy))
    if mag_u <= 1e-9 or mag_v <= 1e-9:
        return 180.0

    dot = (ux * vx + uy * vy) / (mag_u * mag_v)
    dot = max(-1.0, min(1.0, dot))
    return degrees(np.arccos(dot))


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
