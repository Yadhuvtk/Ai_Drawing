from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import Literal, TypeAlias


LoopPolarity = Literal["positive", "negative"]


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class BoundingBox:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y


@dataclass
class Polyline:
    points: list[Point]


@dataclass
class ClosedContour:
    contour_id: str
    points: list[Point]
    area: float
    bbox: BoundingBox
    is_hole: bool = False
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)


@dataclass
class ContourRegion:
    region_id: str
    outer: ClosedContour
    holes: list[ClosedContour] = field(default_factory=list)


@dataclass(frozen=True)
class SegmentLine:
    start: Point
    end: Point


@dataclass(frozen=True)
class SegmentArcCircular:
    start: Point
    end: Point
    radius: float
    large_arc: bool = False
    sweep: bool = True


@dataclass(frozen=True)
class SegmentArcElliptical:
    start: Point
    end: Point
    radius_x: float
    radius_y: float
    rotation_degrees: float = 0.0
    large_arc: bool = False
    sweep: bool = True


@dataclass(frozen=True)
class SegmentBezierQuadratic:
    start: Point
    control: Point
    end: Point


@dataclass(frozen=True)
class SegmentBezierCubic:
    start: Point
    control1: Point
    control2: Point
    end: Point


Segment: TypeAlias = (
    SegmentLine
    | SegmentArcCircular
    | SegmentArcElliptical
    | SegmentBezierQuadratic
    | SegmentBezierCubic
)


@dataclass(frozen=True)
class PrimitiveCircle:
    center: Point
    radius: float


@dataclass(frozen=True)
class PrimitiveEllipse:
    center: Point
    radius_x: float
    radius_y: float
    rotation_degrees: float = 0.0


@dataclass(frozen=True)
class PrimitiveRectangle:
    center: Point
    width: float
    height: float
    rotation_degrees: float = 0.0


@dataclass(frozen=True)
class PrimitiveRoundedRectangle:
    center: Point
    width: float
    height: float
    corner_radius: float
    rotation_degrees: float = 0.0


LoopPrimitive: TypeAlias = PrimitiveCircle | PrimitiveEllipse | PrimitiveRectangle | PrimitiveRoundedRectangle


@dataclass
class Loop:
    loop_id: str
    segments: list[Segment]
    polarity: LoopPolarity = "positive"
    closed: bool = True
    source_contour_id: str | None = None
    primitive: LoopPrimitive | None = None
    confidence: float = 0.0

    @property
    def contour_id(self) -> str:
        return self.loop_id

    @property
    def primitives(self) -> list[Segment]:
        return self.segments


@dataclass
class Shape:
    shape_id: str
    outer_loop: Loop
    negative_loops: list[Loop] = field(default_factory=list)
    fill: str = "#000000"
    stroke: str | None = None
    layer_id: str | None = None
    z_index: int = 0

    @property
    def outer(self) -> Loop:
        return self.outer_loop

    @property
    def holes(self) -> list[Loop]:
        return self.negative_loops


VectorShape = Shape
FittedContour = Loop
PrimitiveLine = SegmentLine
PrimitiveArc = SegmentArcElliptical
PrimitiveBezier = SegmentBezierCubic
Primitive = Segment


@dataclass
class VectorLayer:
    layer_id: str
    shapes: list[Shape]
    fill: str | None = None
    z_index: int = 0


@dataclass
class VectorDocument:
    width: int
    height: int
    shapes: list[Shape]
    layers: list[VectorLayer] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)


def distance(a: Point, b: Point) -> float:
    return hypot(a.x - b.x, a.y - b.y)


def bounding_box_from_points(points: list[Point]) -> BoundingBox:
    xs = [point.x for point in points]
    ys = [point.y for point in points]
    return BoundingBox(min(xs), min(ys), max(xs), max(ys))


def polygon_area(points: list[Point]) -> float:
    if len(points) < 3:
        return 0.0

    area = 0.0
    for idx, point in enumerate(points):
        nxt = points[(idx + 1) % len(points)]
        area += point.x * nxt.y
        area -= nxt.x * point.y
    return area * 0.5


def polygon_perimeter(points: list[Point]) -> float:
    if len(points) < 2:
        return 0.0

    perimeter = 0.0
    for idx, point in enumerate(points):
        nxt = points[(idx + 1) % len(points)]
        perimeter += distance(point, nxt)
    return perimeter
