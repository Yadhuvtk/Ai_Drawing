from __future__ import annotations

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
    SegmentBezierQuadratic,
    SegmentLine,
    Shape,
    VectorLayer,
    VectorDocument,
)


def export_svg(document: VectorDocument, background: str | None = None) -> str:
    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {document.width} {document.height}'>"
    ]
    if background:
        lines.append(
            f"  <rect x='0' y='0' width='{document.width}' height='{document.height}' fill='{background}'/>"
        )

    if document.layers:
        for layer in sorted(document.layers, key=lambda item: item.z_index):
            lines.extend(_layer_to_svg_lines(layer))
    else:
        for shape in document.shapes:
            for markup in _shape_to_svg_markups(shape):
                lines.append(f"  {markup}")

    lines.append("</svg>")
    return "\n".join(lines)


def _layer_to_svg_lines(layer: VectorLayer) -> list[str]:
    lines = [f"  <g id='{layer.layer_id}' data-fill='{layer.fill or ''}'>"]
    for shape in layer.shapes:
        for markup in _shape_to_svg_markups(shape):
            lines.append(f"    {markup}")
    lines.append("  </g>")
    return lines


def _shape_to_svg_markups(shape: Shape) -> list[str]:
    if not shape.negative_loops:
        primitive_markup = _shape_to_primitive_markup(shape)
        if primitive_markup is not None:
            return [primitive_markup]

        path_data, _ = _loop_to_path_data(shape.outer_loop)
        stroke_attr = shape.stroke if shape.stroke is not None else "none"
        return [f"<path d='{path_data}' fill='{shape.fill}' stroke='{stroke_attr}'/>"]

    outer_markup = _outer_loop_markup(shape)
    hole_markups = [_hole_loop_markup(loop) for loop in shape.negative_loops]
    return [outer_markup, *hole_markups]


def _shape_to_path_data(shape: Shape) -> str:
    parts: list[str] = []
    current: Point | None = None
    for loop in [shape.outer_loop, *shape.negative_loops]:
        loop_data, current = _loop_to_path_data(loop, current)
        if loop_data:
            parts.append(loop_data)
    return " ".join(part for part in parts if part)


def _outer_loop_markup(shape: Shape) -> str:
    primitive_markup = _shape_to_primitive_markup(shape)
    if primitive_markup is not None:
        return primitive_markup

    path_data, _ = _loop_to_path_data(shape.outer_loop)
    stroke_attr = shape.stroke if shape.stroke is not None else "none"
    return f"<path d='{path_data}' fill='{shape.fill or '#000000'}' stroke='{stroke_attr}'/>"


def _hole_loop_markup(loop: Loop) -> str:
    path_data, _ = _loop_to_path_data(loop)
    return f"<path d='{path_data}' fill='#F2F2F2' stroke='none'/>"


def _shape_to_primitive_markup(shape: Shape) -> str | None:
    if shape.negative_loops:
        return None

    stroke_attr = shape.stroke if shape.stroke is not None else "none"
    return _loop_to_primitive_markup(shape.outer_loop, fill=shape.fill, stroke=stroke_attr)


def _loop_to_primitive_markup(loop: Loop, fill: str, stroke: str) -> str | None:
    primitive = loop.primitive
    if primitive is None:
        return None
    if isinstance(primitive, PrimitiveCircle):
        return (
            f"<circle cx='{_fmt_number(primitive.center.x)}' cy='{_fmt_number(primitive.center.y)}' "
            f"r='{_fmt_number(primitive.radius)}' fill='{fill}' stroke='{stroke}'/>"
        )
    if isinstance(primitive, PrimitiveEllipse):
        if abs(primitive.rotation_degrees) < 1e-3:
            return (
                f"<ellipse cx='{_fmt_number(primitive.center.x)}' cy='{_fmt_number(primitive.center.y)}' "
                f"rx='{_fmt_number(primitive.radius_x)}' ry='{_fmt_number(primitive.radius_y)}' "
                f"fill='{fill}' stroke='{stroke}'/>"
            )
        return (
            f"<ellipse cx='{_fmt_number(primitive.center.x)}' cy='{_fmt_number(primitive.center.y)}' "
            f"rx='{_fmt_number(primitive.radius_x)}' ry='{_fmt_number(primitive.radius_y)}' "
            f"transform='rotate({_fmt_number(primitive.rotation_degrees)} {_fmt_number(primitive.center.x)} {_fmt_number(primitive.center.y)})' "
            f"fill='{fill}' stroke='{stroke}'/>"
        )
    if isinstance(primitive, PrimitiveRectangle):
        x = primitive.center.x - primitive.width * 0.5
        y = primitive.center.y - primitive.height * 0.5
        transform = ""
        if abs(primitive.rotation_degrees) >= 1e-3:
            transform = (
                f" transform='rotate({_fmt_number(primitive.rotation_degrees)} "
                f"{_fmt_number(primitive.center.x)} {_fmt_number(primitive.center.y)})'"
            )
        return (
            f"<rect x='{_fmt_number(x)}' y='{_fmt_number(y)}' "
            f"width='{_fmt_number(primitive.width)}' height='{_fmt_number(primitive.height)}'"
            f"{transform} fill='{fill}' stroke='{stroke}'/>"
        )
    if isinstance(primitive, PrimitiveRoundedRectangle):
        x = primitive.center.x - primitive.width * 0.5
        y = primitive.center.y - primitive.height * 0.5
        transform = ""
        if abs(primitive.rotation_degrees) >= 1e-3:
            transform = (
                f" transform='rotate({_fmt_number(primitive.rotation_degrees)} "
                f"{_fmt_number(primitive.center.x)} {_fmt_number(primitive.center.y)})'"
            )
        return (
            f"<rect x='{_fmt_number(x)}' y='{_fmt_number(y)}' "
            f"width='{_fmt_number(primitive.width)}' height='{_fmt_number(primitive.height)}' "
            f"rx='{_fmt_number(primitive.corner_radius)}' ry='{_fmt_number(primitive.corner_radius)}'"
            f"{transform} fill='{fill}' stroke='{stroke}'/>"
        )
    return None


def _loop_to_path_data(loop: Loop, current: Point | None = None) -> tuple[str, Point | None]:
    if not loop.segments:
        return "", current

    commands: list[str] = []
    current_point = current
    subpath_start: Point | None = None
    for index, segment in enumerate(loop.segments):
        start = _segment_start(segment)
        if index == 0 or current_point is None or not _same_point(current_point, start):
            origin = current_point if current_point is not None else Point(0.0, 0.0)
            commands.append(f"m {_fmt_delta(origin, start)}")
            current_point = start
            subpath_start = start
        elif subpath_start is None:
            subpath_start = start

        if isinstance(segment, SegmentLine):
            commands.append(f"l {_fmt_delta(current_point, segment.end)}")
            current_point = segment.end
        elif isinstance(segment, SegmentArcCircular):
            commands.append(
                "a "
                f"{_fmt_number(segment.radius)} {_fmt_number(segment.radius)} 0 "
                f"{int(segment.large_arc)} {int(segment.sweep)} {_fmt_delta(current_point, segment.end)}"
            )
            current_point = segment.end
        elif isinstance(segment, SegmentArcElliptical):
            commands.append(
                "a "
                f"{_fmt_number(segment.radius_x)} {_fmt_number(segment.radius_y)} "
                f"{_fmt_number(segment.rotation_degrees)} "
                f"{int(segment.large_arc)} {int(segment.sweep)} {_fmt_delta(current_point, segment.end)}"
            )
            current_point = segment.end
        elif isinstance(segment, SegmentBezierQuadratic):
            commands.append(
                f"q {_fmt_delta(current_point, segment.control)} {_fmt_delta(current_point, segment.end)}"
            )
            current_point = segment.end
        elif isinstance(segment, SegmentBezierCubic):
            commands.append(
                f"c {_fmt_delta(current_point, segment.control1)} "
                f"{_fmt_delta(current_point, segment.control2)} "
                f"{_fmt_delta(current_point, segment.end)}"
            )
            current_point = segment.end

    if loop.closed:
        commands.append("z")
        current_point = subpath_start
    return " ".join(commands), current_point


def _segment_start(segment: object) -> Point:
    return getattr(segment, "start")


def _fmt_point(point: Point) -> str:
    return f"{_fmt_number(point.x)} {_fmt_number(point.y)}"


def _fmt_delta(start: Point, end: Point) -> str:
    return f"{_fmt_number(end.x - start.x)} {_fmt_number(end.y - start.y)}"


def _fmt_number(value: float) -> str:
    rounded = round(float(value), 2)
    if abs(rounded) < 0.005:
        rounded = 0.0
    text = f"{rounded:.2f}"
    text = text.rstrip("0").rstrip(".")
    return text or "0"


def _same_point(left: Point, right: Point) -> bool:
    return abs(left.x - right.x) <= 1e-9 and abs(left.y - right.y) <= 1e-9
