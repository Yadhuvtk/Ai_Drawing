from __future__ import annotations

from dataclasses import replace
from math import cos, radians, sin

from yd_vector.hybrid_vectorizer.geometry import (
    Loop,
    Point,
    PrimitiveCircle,
    PrimitiveEllipse,
    PrimitiveRectangle,
    PrimitiveRoundedRectangle,
    SegmentArcCircular,
    SegmentArcElliptical,
    SegmentBezierCubic,
    SegmentLine,
)


def build_polyline_loop(
    loop_id: str,
    points: list[Point],
    polarity: str = "positive",
    source_contour_id: str | None = None,
    confidence: float = 0.0,
) -> Loop:
    if len(points) < 2:
        return Loop(
            loop_id=loop_id,
            segments=[],
            polarity=polarity,
            closed=True,
            source_contour_id=source_contour_id,
            confidence=confidence,
        )

    segments = [
        SegmentLine(start=points[index], end=points[(index + 1) % len(points)])
        for index in range(len(points))
    ]
    return Loop(
        loop_id=loop_id,
        segments=segments,
        polarity=polarity,
        closed=True,
        source_contour_id=source_contour_id,
        confidence=confidence,
    )


def build_circle_loop(
    loop_id: str,
    circle: PrimitiveCircle,
    polarity: str = "positive",
    source_contour_id: str | None = None,
    confidence: float = 1.0,
) -> Loop:
    cx = circle.center.x
    cy = circle.center.y
    r = circle.radius
    start = Point(cx + r, cy)
    mid = Point(cx - r, cy)
    segments = [
        SegmentArcCircular(start=start, end=mid, radius=r, large_arc=True, sweep=False),
        SegmentArcCircular(start=mid, end=start, radius=r, large_arc=True, sweep=False),
    ]
    return Loop(
        loop_id=loop_id,
        segments=segments,
        polarity=polarity,
        closed=True,
        source_contour_id=source_contour_id,
        primitive=circle,
        confidence=confidence,
    )


def build_circle_cubic_loop(
    loop_id: str,
    circle: PrimitiveCircle,
    polarity: str = "positive",
    source_contour_id: str | None = None,
    confidence: float = 1.0,
) -> Loop:
    cx = circle.center.x
    cy = circle.center.y
    r = circle.radius
    k = r * 0.5523
    start = Point(cx + r, cy)
    quarter_1_end = Point(cx, cy + r)
    quarter_2_end = Point(cx - r, cy)
    quarter_3_end = Point(cx, cy - r)
    quarter_4_end = start

    segments = [
        SegmentBezierCubic(
            start=start,
            control1=Point(cx + r, cy + k),
            control2=Point(cx + k, cy + r),
            end=quarter_1_end,
        ),
        SegmentBezierCubic(
            start=quarter_1_end,
            control1=Point(cx - k, cy + r),
            control2=Point(cx - r, cy + k),
            end=quarter_2_end,
        ),
        SegmentBezierCubic(
            start=quarter_2_end,
            control1=Point(cx - r, cy - k),
            control2=Point(cx - k, cy - r),
            end=quarter_3_end,
        ),
        SegmentBezierCubic(
            start=quarter_3_end,
            control1=Point(cx + k, cy - r),
            control2=Point(cx + r, cy - k),
            end=quarter_4_end,
        ),
    ]
    return Loop(
        loop_id=loop_id,
        segments=segments,
        polarity=polarity,
        closed=True,
        source_contour_id=source_contour_id,
        confidence=confidence,
    )


def build_ellipse_loop(
    loop_id: str,
    ellipse: PrimitiveEllipse,
    polarity: str = "positive",
    source_contour_id: str | None = None,
    confidence: float = 1.0,
) -> Loop:
    start = _rotate_point(ellipse.center, ellipse.rotation_degrees, ellipse.radius_x, 0.0)
    mid = _rotate_point(ellipse.center, ellipse.rotation_degrees, -ellipse.radius_x, 0.0)
    segments = [
        SegmentArcElliptical(
            start=start,
            end=mid,
            radius_x=ellipse.radius_x,
            radius_y=ellipse.radius_y,
            rotation_degrees=ellipse.rotation_degrees,
            large_arc=True,
            sweep=False,
        ),
        SegmentArcElliptical(
            start=mid,
            end=start,
            radius_x=ellipse.radius_x,
            radius_y=ellipse.radius_y,
            rotation_degrees=ellipse.rotation_degrees,
            large_arc=True,
            sweep=False,
        ),
    ]
    return Loop(
        loop_id=loop_id,
        segments=segments,
        polarity=polarity,
        closed=True,
        source_contour_id=source_contour_id,
        primitive=ellipse,
        confidence=confidence,
    )


def build_rectangle_loop(
    loop_id: str,
    rectangle: PrimitiveRectangle,
    polarity: str = "positive",
    source_contour_id: str | None = None,
    confidence: float = 1.0,
) -> Loop:
    hw = rectangle.width * 0.5
    hh = rectangle.height * 0.5
    corners = [
        _rotate_point(rectangle.center, rectangle.rotation_degrees, -hw, -hh),
        _rotate_point(rectangle.center, rectangle.rotation_degrees, hw, -hh),
        _rotate_point(rectangle.center, rectangle.rotation_degrees, hw, hh),
        _rotate_point(rectangle.center, rectangle.rotation_degrees, -hw, hh),
    ]
    segments = [
        SegmentLine(start=corners[index], end=corners[(index + 1) % 4])
        for index in range(4)
    ]
    return Loop(
        loop_id=loop_id,
        segments=segments,
        polarity=polarity,
        closed=True,
        source_contour_id=source_contour_id,
        primitive=rectangle,
        confidence=confidence,
    )


def build_rounded_rectangle_loop(
    loop_id: str,
    rounded_rectangle: PrimitiveRoundedRectangle,
    polarity: str = "positive",
    source_contour_id: str | None = None,
    confidence: float = 1.0,
) -> Loop:
    radius = min(
        rounded_rectangle.corner_radius,
        rounded_rectangle.width * 0.5,
        rounded_rectangle.height * 0.5,
    )
    if radius <= 1e-3:
        rectangle = PrimitiveRectangle(
            center=rounded_rectangle.center,
            width=rounded_rectangle.width,
            height=rounded_rectangle.height,
            rotation_degrees=rounded_rectangle.rotation_degrees,
        )
        loop = build_rectangle_loop(
            loop_id=loop_id,
            rectangle=rectangle,
            polarity=polarity,
            source_contour_id=source_contour_id,
            confidence=confidence,
        )
        return replace(loop, primitive=rounded_rectangle)

    hw = rounded_rectangle.width * 0.5
    hh = rounded_rectangle.height * 0.5
    cx = rounded_rectangle.center.x
    cy = rounded_rectangle.center.y
    rot = rounded_rectangle.rotation_degrees

    top_left = _rotate_point(rounded_rectangle.center, rot, -hw + radius, -hh)
    top_right = _rotate_point(rounded_rectangle.center, rot, hw - radius, -hh)
    right_top = _rotate_point(rounded_rectangle.center, rot, hw, -hh + radius)
    right_bottom = _rotate_point(rounded_rectangle.center, rot, hw, hh - radius)
    bottom_right = _rotate_point(rounded_rectangle.center, rot, hw - radius, hh)
    bottom_left = _rotate_point(rounded_rectangle.center, rot, -hw + radius, hh)
    left_bottom = _rotate_point(rounded_rectangle.center, rot, -hw, hh - radius)
    left_top = _rotate_point(rounded_rectangle.center, rot, -hw, -hh + radius)

    segments = [
        SegmentLine(start=top_left, end=top_right),
        SegmentArcCircular(start=top_right, end=right_top, radius=radius, large_arc=False, sweep=True),
        SegmentLine(start=right_top, end=right_bottom),
        SegmentArcCircular(start=right_bottom, end=bottom_right, radius=radius, large_arc=False, sweep=True),
        SegmentLine(start=bottom_right, end=bottom_left),
        SegmentArcCircular(start=bottom_left, end=left_bottom, radius=radius, large_arc=False, sweep=True),
        SegmentLine(start=left_bottom, end=left_top),
        SegmentArcCircular(start=left_top, end=top_left, radius=radius, large_arc=False, sweep=True),
    ]
    return Loop(
        loop_id=loop_id,
        segments=segments,
        polarity=polarity,
        closed=True,
        source_contour_id=source_contour_id,
        primitive=rounded_rectangle,
        confidence=confidence,
    )


def _rotate_point(center: Point, rotation_degrees: float, local_x: float, local_y: float) -> Point:
    angle = radians(rotation_degrees)
    rotated_x = local_x * cos(angle) - local_y * sin(angle)
    rotated_y = local_x * sin(angle) + local_y * cos(angle)
    return Point(center.x + rotated_x, center.y + rotated_y)
