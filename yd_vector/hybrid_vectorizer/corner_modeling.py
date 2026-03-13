from __future__ import annotations

from dataclasses import dataclass
from math import acos, degrees, tan
from typing import Literal

from yd_vector.hybrid_vectorizer.config import HybridVectorizerConfig
from yd_vector.hybrid_vectorizer.geometry import Point, PrimitiveBezier, SegmentArcCircular, distance


CornerClass = Literal["preserve_sharp", "apply_small_fillet", "treat_as_smooth_curve"]


@dataclass(frozen=True)
class CornerTransition:
    entry_point: Point
    control1: Point
    control2: Point
    exit_point: Point
    radius: float
    sweep: bool

    def as_bezier(self) -> PrimitiveBezier:
        return PrimitiveBezier(
            start=self.entry_point,
            control1=self.control1,
            control2=self.control2,
            end=self.exit_point,
        )

    def as_arc(self) -> SegmentArcCircular:
        return SegmentArcCircular(
            start=self.entry_point,
            end=self.exit_point,
            radius=self.radius,
            large_arc=False,
            sweep=self.sweep,
        )


@dataclass(frozen=True)
class CornerCandidate:
    index: int
    angle_degrees: float
    prev_length: float
    next_length: float
    classification: CornerClass
    fillet: CornerTransition | None = None


def classify_contour_corners(
    points: list[Point],
    config: HybridVectorizerConfig,
) -> dict[int, CornerCandidate]:
    if len(points) < 3:
        return {}

    candidates: dict[int, CornerCandidate] = {}
    fillet_threshold = max(config.min_corner_angle_deg, config.sharp_corner_angle_deg)
    smooth_threshold = max(fillet_threshold, config.smooth_curve_angle_threshold)

    for index, point in enumerate(points):
        prev_point = points[index - 1]
        next_point = points[(index + 1) % len(points)]
        angle = corner_angle_degrees(prev_point, point, next_point)
        prev_length = distance(prev_point, point)
        next_length = distance(point, next_point)

        if angle > smooth_threshold:
            continue

        classification: CornerClass = "treat_as_smooth_curve"
        fillet: CornerTransition | None = None
        length_balance = min(prev_length, next_length) / max(1e-6, max(prev_length, next_length))

        if angle <= config.sharp_corner_angle_deg and config.preserve_tip_points:
            classification = "preserve_sharp"
        elif angle <= fillet_threshold:
            if length_balance < 0.58 and config.preserve_tip_points:
                classification = "preserve_sharp"
            else:
                fillet = build_corner_fillet(prev_point, point, next_point, angle, config)
                if fillet is not None:
                    classification = "apply_small_fillet"
                else:
                    classification = "preserve_sharp" if angle <= config.sharp_corner_angle_deg + 10.0 else "treat_as_smooth_curve"

        candidates[index] = CornerCandidate(
            index=index,
            angle_degrees=angle,
            prev_length=prev_length,
            next_length=next_length,
            classification=classification,
            fillet=fillet,
        )
    return candidates


def corner_angle_degrees(prev_point: Point, point: Point, next_point: Point) -> float:
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


def build_corner_fillet(
    prev_point: Point,
    point: Point,
    next_point: Point,
    angle_degrees: float,
    config: HybridVectorizerConfig,
) -> CornerTransition | None:
    angle_radians = angle_degrees * 3.141592653589793 / 180.0
    if angle_radians <= 0.0 or angle_radians >= 3.141592653589793 - 1e-3:
        return None

    prev_length = distance(prev_point, point)
    next_length = distance(point, next_point)
    min_length = min(prev_length, next_length)
    if min_length <= 1.0:
        return None
    max_length = max(prev_length, next_length)
    length_balance = min_length / max(1e-6, max_length)
    if length_balance < 0.58:
        return None

    tangent_factor = tan(angle_radians * 0.5)
    if tangent_factor <= 1e-4:
        return None

    max_trim = min_length * 0.24
    max_radius_from_segments = max_trim * tangent_factor * 0.9
    if config.min_corner_angle_deg <= config.sharp_corner_angle_deg:
        angle_scale = 1.0
    else:
        angle_scale = (angle_degrees - config.sharp_corner_angle_deg) / (
            config.min_corner_angle_deg - config.sharp_corner_angle_deg
        )
        angle_scale = max(0.25, min(1.0, angle_scale))
    radius = min(
        config.max_fillet_radius,
        min_length * config.fillet_radius_ratio * angle_scale,
        max_radius_from_segments,
    )
    if radius <= 0.2:
        return None

    trim = radius / tangent_factor
    if trim <= 0.2 or trim >= max_trim + 1e-6:
        return None

    prev_to_corner = _unit(point.x - prev_point.x, point.y - prev_point.y)
    corner_to_prev = _unit(prev_point.x - point.x, prev_point.y - point.y)
    corner_to_next = _unit(next_point.x - point.x, next_point.y - point.y)
    if prev_to_corner is None or corner_to_prev is None or corner_to_next is None:
        return None

    entry = Point(point.x + corner_to_prev[0] * trim, point.y + corner_to_prev[1] * trim)
    exit = Point(point.x + corner_to_next[0] * trim, point.y + corner_to_next[1] * trim)
    if distance(entry, exit) <= 0.5:
        return None

    turn_radians = max(1e-4, 3.141592653589793 - angle_radians)
    handle = (4.0 / 3.0) * tan(turn_radians * 0.25) * radius
    control1 = Point(entry.x + prev_to_corner[0] * handle, entry.y + prev_to_corner[1] * handle)
    control2 = Point(exit.x - corner_to_next[0] * handle, exit.y - corner_to_next[1] * handle)
    sweep = ((point.x - entry.x) * (exit.y - point.y) - (point.y - entry.y) * (exit.x - point.x)) > 0.0
    return CornerTransition(
        entry_point=entry,
        control1=control1,
        control2=control2,
        exit_point=exit,
        radius=radius,
        sweep=sweep,
    )


def _unit(x: float, y: float) -> tuple[float, float] | None:
    mag = (x * x + y * y) ** 0.5
    if mag <= 1e-6:
        return None
    return (x / mag, y / mag)
