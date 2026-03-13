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
            lines.append(f"  {_shape_to_svg_markup(shape)}")

    lines.append("</svg>")
    return "\n".join(lines)


def _layer_to_svg_lines(layer: VectorLayer) -> list[str]:
    lines = [f"  <g id='{layer.layer_id}' data-fill='{layer.fill or ''}'>"]
    for shape in layer.shapes:
        lines.append(f"    {_shape_to_svg_markup(shape)}")
    lines.append("  </g>")
    return lines


def _shape_to_svg_markup(shape: Shape) -> str:
    primitive_markup = _shape_to_primitive_markup(shape)
    if primitive_markup is not None:
        return primitive_markup

    path_data = _shape_to_path_data(shape)
    stroke_attr = shape.stroke if shape.stroke is not None else "none"
    return f"<path d='{path_data}' fill='{shape.fill}' stroke='{stroke_attr}' fill-rule='evenodd'/>"


def _shape_to_path_data(shape: Shape) -> str:
    parts = [_loop_to_path_data(shape.outer_loop)]
    parts.extend(_loop_to_path_data(loop) for loop in shape.negative_loops)
    return " ".join(part for part in parts if part)


def _shape_to_primitive_markup(shape: Shape) -> str | None:
    if shape.negative_loops:
        return None

    primitive = shape.outer_loop.primitive
    if primitive is None:
        return None

    stroke_attr = shape.stroke if shape.stroke is not None else "none"
    if isinstance(primitive, PrimitiveCircle):
        return (
            f"<circle cx='{_fmt_number(primitive.center.x)}' cy='{_fmt_number(primitive.center.y)}' "
            f"r='{_fmt_number(primitive.radius)}' fill='{shape.fill}' stroke='{stroke_attr}'/>"
        )
    if isinstance(primitive, PrimitiveEllipse):
        if abs(primitive.rotation_degrees) < 1e-3:
            return (
                f"<ellipse cx='{_fmt_number(primitive.center.x)}' cy='{_fmt_number(primitive.center.y)}' "
                f"rx='{_fmt_number(primitive.radius_x)}' ry='{_fmt_number(primitive.radius_y)}' "
                f"fill='{shape.fill}' stroke='{stroke_attr}'/>"
            )
        return (
            f"<ellipse cx='{_fmt_number(primitive.center.x)}' cy='{_fmt_number(primitive.center.y)}' "
            f"rx='{_fmt_number(primitive.radius_x)}' ry='{_fmt_number(primitive.radius_y)}' "
            f"transform='rotate({_fmt_number(primitive.rotation_degrees)} {_fmt_number(primitive.center.x)} {_fmt_number(primitive.center.y)})' "
            f"fill='{shape.fill}' stroke='{stroke_attr}'/>"
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
            f"{transform} fill='{shape.fill}' stroke='{stroke_attr}'/>"
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
            f"{transform} fill='{shape.fill}' stroke='{stroke_attr}'/>"
        )
    return None


def _loop_to_path_data(loop: Loop) -> str:
    if not loop.segments:
        return ""

    commands: list[str] = []
    current: Point | None = None
    for index, segment in enumerate(loop.segments):
        start = _segment_start(segment)
        if index == 0 or current != start:
            commands.append(f"M {_fmt_point(start)}")

        if isinstance(segment, SegmentLine):
            commands.append(f"L {_fmt_point(segment.end)}")
            current = segment.end
        elif isinstance(segment, SegmentArcCircular):
            commands.append(
                "A "
                f"{_fmt_number(segment.radius)} {_fmt_number(segment.radius)} 0 "
                f"{int(segment.large_arc)} {int(segment.sweep)} {_fmt_point(segment.end)}"
            )
            current = segment.end
        elif isinstance(segment, SegmentArcElliptical):
            commands.append(
                "A "
                f"{_fmt_number(segment.radius_x)} {_fmt_number(segment.radius_y)} "
                f"{_fmt_number(segment.rotation_degrees)} "
                f"{int(segment.large_arc)} {int(segment.sweep)} {_fmt_point(segment.end)}"
            )
            current = segment.end
        elif isinstance(segment, SegmentBezierQuadratic):
            commands.append(f"Q {_fmt_point(segment.control)} {_fmt_point(segment.end)}")
            current = segment.end
        elif isinstance(segment, SegmentBezierCubic):
            commands.append(
                f"C {_fmt_point(segment.control1)} {_fmt_point(segment.control2)} {_fmt_point(segment.end)}"
            )
            current = segment.end

    if loop.closed:
        commands.append("Z")
    return " ".join(commands)


def _segment_start(segment: object) -> Point:
    return getattr(segment, "start")


def _fmt_point(point: Point) -> str:
    return f"{_fmt_number(point.x)} {_fmt_number(point.y)}"


def _fmt_number(value: float) -> str:
    rounded = round(float(value), 3)
    text = f"{rounded:.3f}"
    text = text.rstrip("0").rstrip(".")
    return text or "0"
