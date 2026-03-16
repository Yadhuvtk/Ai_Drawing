from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from yd_vector.hybrid_vectorizer.cleanup import (
    detect_hard_corner_indices,
    douglas_peucker_closed,
    douglas_peucker_open,
    merge_near_duplicate_points,
    simplify_closed_preserving_indices,
)
from yd_vector.hybrid_vectorizer.config import HybridVectorizerConfig
from yd_vector.hybrid_vectorizer.corner_modeling import CornerCandidate, classify_contour_corners
from yd_vector.hybrid_vectorizer.cutout_assembly import assemble_shape
from yd_vector.hybrid_vectorizer.fit_curves import fit_curve
from yd_vector.hybrid_vectorizer.geometry import (
    ClosedContour,
    ContourRegion,
    Loop,
    Point,
    PrimitiveCircle,
    PrimitiveEllipse,
    PrimitiveRoundedRectangle,
    Segment,
    SegmentArcCircular,
    SegmentArcElliptical,
    SegmentBezierCubic,
    SegmentBezierQuadratic,
    SegmentLine,
    Shape,
)
from yd_vector.hybrid_vectorizer.loop_builder import (
    build_circle_loop,
    build_circle_cubic_loop,
    build_ellipse_loop,
    build_polyline_loop,
    build_rectangle_loop,
    build_rounded_rectangle_loop,
)
from yd_vector.hybrid_vectorizer.shape_analysis import (
    CircleFitResult,
    CircularArcFitResult,
    EllipseFitResult,
    EllipticalArcFitResult,
    RectangleFitResult,
    RoundedRectangleFitResult,
    ShapeClassification,
    classify_parameterized_shape,
    fit_circle,
    fit_circular_arc,
    fit_elliptical_arc,
    fit_rectangle,
    fit_rotated_ellipse,
    fit_rounded_rectangle,
)
from yd_vector.hybrid_vectorizer.topology_guard import validate_loop_against_contour, validate_shape_topology


FITCURVES_MIN_ERROR = 2.0
FITCURVES_PRE_RDP_EPSILON = 0.8
FITCURVES_MIN_SEGMENT_LENGTH = 3.0
HARD_CORNER_MIN_TURN_DEGREES = 60.0
FITCURVES_REFINED_ERROR = 1.2
FITCURVES_REFINED_RDP_EPSILON = 0.5
SHORT_SHARP_SPAN_LENGTH = 12.0
PIN_BODY_SPAN_ERROR = 1.9
PIN_BODY_SPAN_RDP_EPSILON = 0.95
PIN_BODY_MIN_SIDE_SEGMENTS = 3
PIN_BODY_MAX_SIDE_SEGMENTS = 3
PIN_BODY_PROFILE_SAMPLES = 16
SHADOW_ARC_ERROR = 3.4
SHADOW_ARC_RDP_EPSILON = 2.15
SHADOW_ARC_MIN_SEGMENTS = 2
SHADOW_ARC_MAX_SEGMENTS = 2
SHADOW_PROFILE_SAMPLES = 14


@dataclass(frozen=True)
class FittedShape:
    region_id: str
    outer_classification: ShapeClassification
    hole_classifications: tuple[ShapeClassification, ...]
    shape: Shape


def fit_region(
    region: ContourRegion,
    config: HybridVectorizerConfig,
    fill_color: str | None = None,
    stroke_color: str | None = None,
    shape_id: str | None = None,
    layer_id: str | None = None,
    z_index: int = 0,
    outer_classification: ShapeClassification | dict[str, object] | None = None,
    hole_classifications: list[ShapeClassification | dict[str, object] | None] | None = None,
) -> Shape:
    fitted = fit_classified_region(
        region,
        config,
        fill_color=fill_color,
        stroke_color=stroke_color,
        shape_id=shape_id,
        layer_id=layer_id,
        z_index=z_index,
        outer_classification=outer_classification,
        hole_classifications=hole_classifications,
    )
    return fitted.shape


def fit_classified_region(
    region: ContourRegion,
    config: HybridVectorizerConfig,
    fill_color: str | None = None,
    stroke_color: str | None = None,
    shape_id: str | None = None,
    layer_id: str | None = None,
    z_index: int = 0,
    outer_classification: ShapeClassification | dict[str, object] | None = None,
    hole_classifications: list[ShapeClassification | dict[str, object] | None] | None = None,
) -> FittedShape:
    outer_loop = fit_contour(
        region.outer,
        config,
        polarity="positive",
        classification=outer_classification,
    )
    resolved_outer = _resolve_shape_classification(outer_classification)

    raw_hole_classifications = _normalize_hole_classifications(region, hole_classifications)
    resolved_holes = [_resolve_shape_classification(item) for item in raw_hole_classifications]
    negative_loops = [
        fit_contour(hole, config, polarity="negative", classification=hole_classification)
        for hole, hole_classification in zip(region.holes, raw_hole_classifications)
    ]
    shape = assemble_shape(
        shape_id=shape_id or region.region_id,
        outer_loop=outer_loop,
        negative_loops=negative_loops,
        fill=fill_color or config.fill_color,
        stroke=stroke_color if stroke_color is not None else config.stroke_color,
        layer_id=layer_id,
        z_index=z_index,
    )
    if validate_shape_topology(shape, region.outer, region.holes, config):
        return FittedShape(
            region_id=region.region_id,
            outer_classification=resolved_outer,
            hole_classifications=tuple(resolved_holes),
            shape=shape,
        )

    fallback = assemble_shape(
        shape_id=shape_id or region.region_id,
        outer_loop=_polyline_loop_from_contour(region.outer, polarity="positive"),
        negative_loops=[_polyline_loop_from_contour(hole, polarity="negative") for hole in region.holes],
        fill=fill_color or config.fill_color,
        stroke=stroke_color if stroke_color is not None else config.stroke_color,
        layer_id=layer_id,
        z_index=z_index,
    )
    return FittedShape(
        region_id=region.region_id,
        outer_classification=resolved_outer,
        hole_classifications=tuple(resolved_holes),
        shape=fallback,
    )


def _resolve_shape_classification(
    classification: ShapeClassification | dict[str, object] | None,
) -> ShapeClassification:
    if isinstance(classification, ShapeClassification):
        return classification
    if isinstance(classification, dict):
        return ShapeClassification(
            representation=str(classification.get("representation", "freeform")),
            kind=str(classification.get("kind", "freeform")),
            confidence=float(classification.get("confidence", 1.0)),
            params=dict(classification.get("params", {})) if isinstance(classification.get("params", {}), dict) else {},
            descriptors=dict(classification.get("descriptors", {})) if isinstance(classification.get("descriptors", {}), dict) else {},
            segments=tuple(classification.get("segments", [])) if isinstance(classification.get("segments", []), list) else (),
        )
    return ShapeClassification(
        representation="freeform",
        kind="freeform",
        confidence=1.0,
        params={},
        descriptors={},
        segments=(),
    )


def _resolve_hole_classifications(
    region: ContourRegion,
    hole_classifications: list[ShapeClassification | dict[str, object] | None] | None,
) -> list[ShapeClassification]:
    return [_resolve_shape_classification(item) for item in _normalize_hole_classifications(region, hole_classifications)]


def _normalize_hole_classifications(
    region: ContourRegion,
    hole_classifications: list[ShapeClassification | dict[str, object] | None] | None,
) -> list[ShapeClassification | dict[str, object] | None]:
    resolved: list[ShapeClassification | dict[str, object] | None] = []
    provided = hole_classifications or []
    for index, _hole in enumerate(region.holes):
        if index < len(provided):
            resolved.append(provided[index])
        else:
            resolved.append(None)
    return resolved


def _classification_to_candidate(
    classification: ShapeClassification | dict[str, object] | None,
) -> dict[str, object] | None:
    resolved = _resolve_shape_classification(classification)
    if resolved.kind == "freeform" or resolved.representation != "parameterized":
        return None
    return {
        "representation": resolved.representation,
        "kind": resolved.kind,
        "confidence": resolved.confidence,
        "params": resolved.params,
        "descriptors": resolved.descriptors,
        "segments": list(resolved.segments),
    }


def fit_contour(
    contour: ClosedContour,
    config: HybridVectorizerConfig,
    polarity: str = "positive",
    classification: ShapeClassification | dict[str, object] | None = None,
) -> Loop:
    parameterized = _fit_parameterized_loop(contour, config, polarity, classification=classification)
    if parameterized is not None and validate_loop_against_contour(parameterized, contour, config):
        return parameterized

    segments = fit_freeform_segments(contour.points, config, contour=contour)
    candidate = Loop(
        loop_id=contour.contour_id,
        segments=segments,
        polarity=polarity,
        closed=True,
        source_contour_id=contour.contour_id,
        confidence=0.0,
    )
    if validate_loop_against_contour(candidate, contour, config):
        return candidate

    smooth_fallback = _fit_safe_spline_loop(contour, config, polarity)
    if smooth_fallback is not None and validate_loop_against_contour(smooth_fallback, contour, config):
        return smooth_fallback
    return _polyline_loop_from_contour(contour, polarity=polarity)


def _fit_specialized_outer_loop(region: ContourRegion, config: HybridVectorizerConfig) -> Loop | None:
    del region, config
    return None


def _should_prefer_specialized_outer_loop(region: ContourRegion, candidate: Loop, baseline: Loop) -> bool:
    del region, candidate, baseline
    return False


def _fit_parameterized_loop(
    contour: ClosedContour,
    config: HybridVectorizerConfig,
    polarity: str,
    classification: ShapeClassification | dict[str, object] | None = None,
) -> Loop | None:
    classified = _classification_to_candidate(classification)
    if classification is not None and classified is None:
        return None
    if classified is None:
        classified = classify_parameterized_shape(
            contour.points,
            metadata={"is_hole": contour.is_hole},
        )
    if classified is not None:
        classified_loop = _build_parameterized_candidate_loop(
            contour=contour,
            candidate=classified,
            polarity=polarity,
        )
        if classified_loop is not None:
            return classified_loop

    candidates: list[tuple[float, Loop]] = []

    circle_result = fit_circle(contour.points)
    if _is_circle_candidate(circle_result, contour, config):
        return build_circle_cubic_loop(
            loop_id=contour.contour_id,
            circle=circle_result.primitive,
            polarity=polarity,
            source_contour_id=contour.contour_id,
            confidence=circle_result.confidence,
        )

    ellipse_result = fit_rotated_ellipse(contour.points)
    if _is_ellipse_candidate(ellipse_result, config):
        candidates.append(
            (
                ellipse_result.confidence + 0.01,
                build_ellipse_loop(
                    loop_id=contour.contour_id,
                    ellipse=ellipse_result.primitive,
                    polarity=polarity,
                    source_contour_id=contour.contour_id,
                    confidence=ellipse_result.confidence,
                ),
            )
        )

    rounded_rect_result = fit_rounded_rectangle(contour.points)
    if _is_rounded_rectangle_candidate(rounded_rect_result, config):
        candidates.append(
            (
                rounded_rect_result.confidence,
                build_rounded_rectangle_loop(
                    loop_id=contour.contour_id,
                    rounded_rectangle=rounded_rect_result.primitive,
                    polarity=polarity,
                    source_contour_id=contour.contour_id,
                    confidence=rounded_rect_result.confidence,
                ),
            )
        )

    rectangle_result = fit_rectangle(contour.points)
    if _is_rectangle_candidate(rectangle_result, config):
        candidates.append(
            (
                rectangle_result.confidence,
                build_rectangle_loop(
                    loop_id=contour.contour_id,
                    rectangle=rectangle_result.primitive,
                    polarity=polarity,
                    source_contour_id=contour.contour_id,
                    confidence=rectangle_result.confidence,
                ),
            )
        )

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _build_parameterized_candidate_loop(
    contour: ClosedContour,
    candidate: dict[str, object],
    polarity: str,
) -> Loop | None:
    kind = str(candidate.get("kind", ""))
    params = candidate.get("params", {})
    if not isinstance(params, dict):
        return None
    confidence = float(candidate.get("confidence", 0.0))

    if kind == "circle":
        primitive = params.get("primitive")
        if primitive is None:
            return None
        return build_circle_loop(
            loop_id=contour.contour_id,
            circle=primitive,
            polarity=polarity,
            source_contour_id=contour.contour_id,
            confidence=confidence,
        )

    if kind == "ellipse":
        primitive = params.get("primitive")
        if primitive is None:
            return None
        return build_ellipse_loop(
            loop_id=contour.contour_id,
            ellipse=primitive,
            polarity=polarity,
            source_contour_id=contour.contour_id,
            confidence=confidence,
        )

    if kind == "rounded_rectangle":
        primitive = params.get("primitive")
        if primitive is None:
            return None
        return build_rounded_rectangle_loop(
            loop_id=contour.contour_id,
            rounded_rectangle=primitive,
            polarity=polarity,
            source_contour_id=contour.contour_id,
            confidence=confidence,
        )

    if kind == "triangle":
        return _build_isosceles_triangle_loop(
            loop_id=contour.contour_id,
            params=params,
            polarity=polarity,
            source_contour_id=contour.contour_id,
            confidence=confidence,
        )

    if kind == "star":
        return _build_star_loop(
            loop_id=contour.contour_id,
            params=params,
            polarity=polarity,
            source_contour_id=contour.contour_id,
            confidence=confidence,
        )

    if kind == "dshape":
        return _build_dshape_loop(
            loop_id=contour.contour_id,
            params=params,
            polarity=polarity,
            source_contour_id=contour.contour_id,
            confidence=confidence,
        )
    return None


def build_circle_segments(params: dict[str, object]) -> list[Segment]:
    primitive = params.get("primitive")
    if not isinstance(primitive, PrimitiveCircle):
        primitive = PrimitiveCircle(
            center=Point(float(params["cx"]), float(params["cy"])),
            radius=float(params["r"]),
        )
    return build_circle_loop(loop_id="circle_builder", circle=primitive).segments


def build_ellipse_segments(params: dict[str, object]) -> list[Segment]:
    primitive = params.get("primitive")
    if not isinstance(primitive, PrimitiveEllipse):
        primitive = PrimitiveEllipse(
            center=Point(float(params["cx"]), float(params["cy"])),
            radius_x=float(params["rx"]),
            radius_y=float(params["ry"]),
            rotation_degrees=float(params.get("theta", 0.0)),
        )
    return build_ellipse_loop(loop_id="ellipse_builder", ellipse=primitive).segments


def build_rounded_rectangle_segments(params: dict[str, object]) -> list[Segment]:
    primitive = params.get("primitive")
    if not isinstance(primitive, PrimitiveRoundedRectangle):
        primitive = PrimitiveRoundedRectangle(
            center=Point(float(params["cx"]), float(params["cy"])),
            width=float(params["w"]),
            height=float(params["h"]),
            corner_radius=float(params.get("corner_radius", 0.0)),
            rotation_degrees=float(params.get("theta", 0.0)),
        )
    return build_rounded_rectangle_loop(loop_id="rounded_rectangle_builder", rounded_rectangle=primitive).segments


def build_triangle_segments(params: dict[str, object]) -> list[Segment]:
    loop = _build_isosceles_triangle_loop(
        loop_id="triangle_builder",
        params=params,
        polarity="positive",
        source_contour_id=None,
        confidence=1.0,
    )
    return [] if loop is None else loop.segments


def build_star_segments(params: dict[str, object]) -> list[Segment]:
    loop = _build_star_loop(
        loop_id="star_builder",
        params=params,
        polarity="positive",
        source_contour_id=None,
        confidence=1.0,
    )
    return [] if loop is None else loop.segments


def build_capsule_or_dshape_segments(params: dict[str, object]) -> list[Segment]:
    loop = _build_dshape_loop(
        loop_id="dshape_builder",
        params=params,
        polarity="positive",
        source_contour_id=None,
        confidence=1.0,
    )
    return [] if loop is None else loop.segments


def build_freeform_segments(
    points: list[Point],
    preserved_corners: set[int] | list[int] | None = None,
    config: HybridVectorizerConfig | None = None,
) -> list[Segment]:
    del preserved_corners  # TODO: plumb explicit preserved-corner spans through this public wrapper.
    return fit_freeform_segments(points, config or HybridVectorizerConfig())


def _build_isosceles_triangle_loop(
    loop_id: str,
    params: dict[str, object],
    polarity: str,
    source_contour_id: str | None,
    confidence: float,
) -> Loop | None:
    vertices = params.get("vertices")
    if not isinstance(vertices, list) or len(vertices) != 3:
        return None
    triangle_points = [vertex for vertex in vertices if isinstance(vertex, Point)]
    if len(triangle_points) != 3:
        return None
    return build_polyline_loop(
        loop_id=loop_id,
        points=triangle_points,
        polarity=polarity,
        source_contour_id=source_contour_id,
        confidence=confidence,
    )


def _build_star_loop(
    loop_id: str,
    params: dict[str, object],
    polarity: str,
    source_contour_id: str | None,
    confidence: float,
) -> Loop | None:
    try:
        cx = float(params["cx"])
        cy = float(params["cy"])
        point_count = int(params["n"])
        r_outer = float(params["r_outer"])
        r_inner = float(params["r_inner"])
        theta_degrees = float(params.get("theta", 0.0))
    except (KeyError, TypeError, ValueError):
        return None

    if point_count < 3 or r_outer <= 0.0 or r_inner <= 0.0 or r_inner >= r_outer:
        return None

    vertices: list[Point] = []
    base_angle = np.deg2rad(theta_degrees)
    step = np.pi / float(point_count)
    for index in range(point_count * 2):
        radius = r_outer if index % 2 == 0 else r_inner
        angle = base_angle + (index * step)
        vertices.append(Point(cx + radius * float(np.cos(angle)), cy + radius * float(np.sin(angle))))

    return build_polyline_loop(
        loop_id=loop_id,
        points=vertices,
        polarity=polarity,
        source_contour_id=source_contour_id,
        confidence=confidence,
    )


def _build_dshape_loop(
    loop_id: str,
    params: dict[str, object],
    polarity: str,
    source_contour_id: str | None,
    confidence: float,
) -> Loop | None:
    del loop_id, params, polarity, source_contour_id, confidence
    # TODO: add a true D-shape loop builder once the detector is validated on stable samples.
    return None


def _is_pin_body_region(region: ContourRegion) -> bool:
    if len(region.holes) != 1:
        return False

    hole_fit = fit_circle(region.holes[0].points)
    if hole_fit is None:
        return False

    outer_bbox = region.outer.bbox
    if outer_bbox.width <= 0.0 or outer_bbox.height <= 0.0:
        return False

    aspect_ratio = outer_bbox.height / outer_bbox.width
    if aspect_ratio < 1.18:
        return False

    if hole_fit.confidence < 0.88 and hole_fit.circularity < 0.9:
        return False

    center_x = outer_bbox.min_x + outer_bbox.width * 0.5
    hole_center = hole_fit.primitive.center
    if abs(hole_center.x - center_x) > outer_bbox.width * 0.11:
        return False
    if not (outer_bbox.min_y + outer_bbox.height * 0.16 <= hole_center.y <= outer_bbox.min_y + outer_bbox.height * 0.54):
        return False

    bottom_point = max(region.outer.points, key=lambda point: point.y)
    return abs(bottom_point.x - center_x) <= outer_bbox.width * 0.14


def _fit_pin_body_outer_loop(
    contour: ClosedContour,
    config: HybridVectorizerConfig,
    symmetry_axis_x: float | None = None,
) -> Loop | None:
    if len(contour.points) < 12:
        return None

    center_x = contour.bbox.min_x + contour.bbox.width * 0.5 if symmetry_axis_x is None else float(symmetry_axis_x)
    top_index = _center_weighted_extreme_index(contour.points, axis="top", center_x=center_x)
    bottom_index = _center_weighted_extreme_index(contour.points, axis="bottom", center_x=center_x)
    if top_index == bottom_index:
        return None

    top_to_bottom = _closed_span(contour.points, top_index, bottom_index)
    bottom_to_top = _closed_span(contour.points, bottom_index, top_index)
    if len(top_to_bottom) < 4 or len(bottom_to_top) < 4:
        return None

    if _mean_x(top_to_bottom) >= _mean_x(bottom_to_top):
        right_span = top_to_bottom
        left_span = list(reversed(bottom_to_top))
    else:
        right_span = list(reversed(bottom_to_top))
        left_span = top_to_bottom

    top_point = Point(center_x, contour.points[top_index].y)
    bottom_point = Point(center_x, contour.points[bottom_index].y)
    right_profile = _build_symmetric_profile_span(
        primary_span=right_span,
        mirrored_span=left_span,
        axis_x=center_x,
        sample_count=_profile_sample_count(
            right_span,
            left_span,
            default=PIN_BODY_PROFILE_SAMPLES,
            minimum=12,
            maximum=26,
        ),
        start_point=top_point,
        end_point=bottom_point,
        symmetry_blend=0.92,
    )
    right_segments = _fit_profiled_open_span(
        right_profile,
        error=max(config.bezier_fit_tolerance, PIN_BODY_SPAN_ERROR),
        rdp_tolerance=PIN_BODY_SPAN_RDP_EPSILON,
        min_segments=PIN_BODY_MIN_SIDE_SEGMENTS,
        max_segments=PIN_BODY_MAX_SIDE_SEGMENTS,
        sigma=1.08,
    )
    if len(right_segments) < PIN_BODY_MIN_SIDE_SEGMENTS and len(right_profile) >= 8:
        right_segments = _fit_profiled_open_span(
            right_profile,
            error=max(config.bezier_fit_tolerance, PIN_BODY_SPAN_ERROR * 0.92),
            rdp_tolerance=max(0.7, PIN_BODY_SPAN_RDP_EPSILON * 0.9),
            min_segments=PIN_BODY_MIN_SIDE_SEGMENTS,
            max_segments=PIN_BODY_MAX_SIDE_SEGMENTS,
            sigma=0.95,
        )
    right_segments = _limit_cubic_segment_count(right_segments, PIN_BODY_MAX_SIDE_SEGMENTS)
    if not right_segments:
        return None
    left_segments = _mirror_segments_across_vertical_axis(right_segments, axis_x=center_x)
    if not left_segments:
        return None

    return Loop(
        loop_id=contour.contour_id,
        segments=right_segments + left_segments,
        polarity="positive",
        closed=True,
        source_contour_id=contour.contour_id,
        confidence=0.0,
    )


def _fit_shadow_outer_loop(
    contour: ClosedContour,
    config: HybridVectorizerConfig,
    symmetry_axis_x: float | None = None,
) -> Loop | None:
    anchor_indices = _detect_shadow_anchor_indices(contour.points, contour)
    if anchor_indices is None:
        return None

    left_bridge_index, notch_index, right_bridge_index = anchor_indices
    left_bridge = contour.points[left_bridge_index]
    notch_point = contour.points[notch_index]
    right_bridge = contour.points[right_bridge_index]

    bottom_span = _select_bottom_shadow_span(
        contour.points,
        right_bridge_index,
        left_bridge_index,
        notch_index,
    )
    if len(bottom_span) < 4:
        return None

    center_x = (left_bridge.x + right_bridge.x) * 0.5 if symmetry_axis_x is None else float(symmetry_axis_x)
    bottom_index = max(
        range(len(bottom_span)),
        key=lambda idx: (bottom_span[idx].y, -abs(bottom_span[idx].x - center_x)),
    )
    if bottom_index <= 0 or bottom_index >= len(bottom_span) - 1:
        return None

    right_half = bottom_span[: bottom_index + 1]
    left_half = list(reversed(bottom_span[bottom_index:]))
    bottom_point = Point(center_x, bottom_span[bottom_index].y)
    right_profile = _build_symmetric_profile_span(
        primary_span=right_half,
        mirrored_span=left_half,
        axis_x=center_x,
        sample_count=_profile_sample_count(
            right_half,
            left_half,
            default=SHADOW_PROFILE_SAMPLES,
            minimum=10,
            maximum=20,
        ),
        start_point=right_bridge,
        end_point=bottom_point,
        symmetry_blend=0.97,
        bridge_y=max(left_bridge.y, right_bridge.y),
        depth_power=1.35,
    )
    arc_segments = _fit_profiled_open_span(
        right_profile,
        error=max(config.bezier_fit_tolerance + 0.75, SHADOW_ARC_ERROR),
        rdp_tolerance=SHADOW_ARC_RDP_EPSILON,
        min_segments=SHADOW_ARC_MIN_SEGMENTS,
        max_segments=SHADOW_ARC_MAX_SEGMENTS,
        sigma=1.3,
    )
    arc_segments = _limit_cubic_segment_count(arc_segments, SHADOW_ARC_MAX_SEGMENTS)
    if not arc_segments:
        return None
    mirrored_arc = _mirror_segments_across_vertical_axis(arc_segments, axis_x=center_x)
    if not mirrored_arc:
        return None

    return Loop(
        loop_id=contour.contour_id,
        segments=[
            SegmentLine(start=left_bridge, end=notch_point),
            SegmentLine(start=notch_point, end=right_bridge),
            *arc_segments,
            *mirrored_arc,
        ],
        polarity="positive",
        closed=True,
        source_contour_id=contour.contour_id,
        confidence=0.0,
    )


def _profile_sample_count(
    primary_span: list[Point],
    mirrored_span: list[Point],
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    base = max(default, int(round(max(len(primary_span), len(mirrored_span)) / 10.0)))
    return max(minimum, min(maximum, base))


def _build_symmetric_profile_span(
    primary_span: list[Point],
    mirrored_span: list[Point],
    axis_x: float,
    sample_count: int,
    start_point: Point,
    end_point: Point,
    symmetry_blend: float,
    bridge_y: float | None = None,
    depth_power: float = 1.0,
) -> list[Point]:
    if len(primary_span) < 2 or len(mirrored_span) < 2:
        return [start_point, end_point]

    primary_samples = _resample_open_span(primary_span, sample_count)
    mirrored_samples = _resample_open_span(mirrored_span, sample_count)
    depth_denominator = max(1.0, end_point.y - (bridge_y if bridge_y is not None else start_point.y))

    profile = [start_point]
    for index in range(1, sample_count - 1):
        primary_point = primary_samples[index]
        mirrored_point = mirrored_samples[index]
        average_y = (primary_point.y + mirrored_point.y) * 0.5
        average_offset = (
            abs(primary_point.x - axis_x) + abs(mirrored_point.x - axis_x)
        ) * 0.5
        symmetric_target = Point(axis_x + average_offset, average_y)

        blend = max(0.0, min(1.0, symmetry_blend))
        if bridge_y is not None:
            depth = max(0.0, min(1.0, (average_y - bridge_y) / depth_denominator))
            blend *= depth**depth_power

        profile.append(
            Point(
                x=primary_point.x + blend * (symmetric_target.x - primary_point.x),
                y=primary_point.y + blend * (symmetric_target.y - primary_point.y),
            )
        )
    profile.append(end_point)
    return merge_near_duplicate_points(profile, merge_distance=0.2, closed=False)


def _resample_open_span(points: list[Point], sample_count: int) -> list[Point]:
    if len(points) <= 2 or sample_count <= 2:
        return [points[0], points[-1]]

    cumulative = [0.0]
    for index in range(1, len(points)):
        cumulative.append(cumulative[-1] + _distance(points[index - 1], points[index]))
    total_length = cumulative[-1]
    if total_length <= 1e-6:
        return [points[0]] + [points[-1]] * (sample_count - 1)

    sampled = [points[0]]
    segment_index = 1
    for target_index in range(1, sample_count - 1):
        target_distance = total_length * (target_index / float(sample_count - 1))
        while segment_index < len(cumulative) - 1 and cumulative[segment_index] < target_distance:
            segment_index += 1
        left_index = max(0, segment_index - 1)
        right_index = min(len(points) - 1, segment_index)
        start_distance = cumulative[left_index]
        end_distance = cumulative[right_index]
        if end_distance - start_distance <= 1e-6:
            sampled.append(points[right_index])
            continue
        t = (target_distance - start_distance) / (end_distance - start_distance)
        start_point = points[left_index]
        end_point = points[right_index]
        sampled.append(
            Point(
                x=start_point.x + (end_point.x - start_point.x) * t,
                y=start_point.y + (end_point.y - start_point.y) * t,
            )
        )
    sampled.append(points[-1])
    return sampled


def _mirror_segments_across_vertical_axis(segments: list[Segment], axis_x: float) -> list[Segment]:
    mirrored: list[Segment] = []
    for segment in reversed(segments):
        mirrored_segment = _mirror_segment_across_vertical_axis(segment, axis_x)
        if mirrored_segment is not None:
            mirrored.append(mirrored_segment)
    return mirrored


def _limit_cubic_segment_count(segments: list[Segment], max_segments: int) -> list[Segment]:
    if max_segments <= 0 or len(segments) <= max_segments:
        return segments
    if not all(isinstance(segment, SegmentBezierCubic) for segment in segments):
        return segments

    current = list(segments)
    while len(current) > max_segments:
        merge_index = min(
            range(len(current) - 1),
            key=lambda index: _distance(current[index].start, current[index].end)
            + _distance(current[index + 1].start, current[index + 1].end),
        )
        left = current[merge_index]
        right = current[merge_index + 1]
        current[merge_index : merge_index + 2] = [
            SegmentBezierCubic(
                start=left.start,
                control1=left.control1,
                control2=right.control2,
                end=right.end,
            )
        ]
    return current


def _mirror_segment_across_vertical_axis(segment: Segment, axis_x: float) -> Segment | None:
    if isinstance(segment, SegmentLine):
        return SegmentLine(
            start=_mirror_point_across_vertical_axis(segment.end, axis_x),
            end=_mirror_point_across_vertical_axis(segment.start, axis_x),
        )
    if isinstance(segment, SegmentBezierQuadratic):
        return SegmentBezierQuadratic(
            start=_mirror_point_across_vertical_axis(segment.end, axis_x),
            control=_mirror_point_across_vertical_axis(segment.control, axis_x),
            end=_mirror_point_across_vertical_axis(segment.start, axis_x),
        )
    if isinstance(segment, SegmentBezierCubic):
        return SegmentBezierCubic(
            start=_mirror_point_across_vertical_axis(segment.end, axis_x),
            control1=_mirror_point_across_vertical_axis(segment.control2, axis_x),
            control2=_mirror_point_across_vertical_axis(segment.control1, axis_x),
            end=_mirror_point_across_vertical_axis(segment.start, axis_x),
        )
    if isinstance(segment, SegmentArcCircular):
        return SegmentArcCircular(
            start=_mirror_point_across_vertical_axis(segment.end, axis_x),
            end=_mirror_point_across_vertical_axis(segment.start, axis_x),
            radius=segment.radius,
            large_arc=segment.large_arc,
            sweep=not segment.sweep,
        )
    if isinstance(segment, SegmentArcElliptical):
        return SegmentArcElliptical(
            start=_mirror_point_across_vertical_axis(segment.end, axis_x),
            end=_mirror_point_across_vertical_axis(segment.start, axis_x),
            radius_x=segment.radius_x,
            radius_y=segment.radius_y,
            rotation_degrees=-segment.rotation_degrees,
            large_arc=segment.large_arc,
            sweep=not segment.sweep,
        )
    return None


def _mirror_point_across_vertical_axis(point: Point, axis_x: float) -> Point:
    return Point(x=(2.0 * axis_x) - point.x, y=point.y)


def _is_circle_candidate(
    result: CircleFitResult | None,
    contour: ClosedContour,
    config: HybridVectorizerConfig,
) -> bool:
    if result is None:
        return False

    bbox = contour.bbox
    aspect_ratio = bbox.width / max(1e-6, bbox.height)
    if result.circularity > 0.88:
        return (
            0.82 <= aspect_ratio <= 1.18
            and result.max_radial_error <= max(0.08, config.circle_fit_tolerance * 1.8)
            and 0.72 <= result.area_ratio <= 1.08
        )
    return (
        0.9 <= aspect_ratio <= 1.1
        and result.radial_error <= config.circle_fit_tolerance
        and result.max_radial_error <= config.circle_fit_tolerance * 1.45
        and result.circularity >= config.circle_circularity_min
        and 0.88 <= result.area_ratio <= 1.12
        and result.confidence >= config.circle_confidence_threshold
    )


def _is_ellipse_candidate(result: EllipseFitResult | None, config: HybridVectorizerConfig) -> bool:
    if result is None:
        return False
    return (
        result.normalized_error <= config.ellipse_fit_tolerance
        and result.max_normalized_error <= config.ellipse_fit_tolerance * 1.5
        and 0.84 <= result.area_ratio <= 1.16
        and result.confidence >= config.ellipse_confidence_threshold
    )


def _is_rectangle_candidate(result: RectangleFitResult | None, config: HybridVectorizerConfig) -> bool:
    if result is None:
        return False
    return (
        result.mean_error <= config.primitive_fit_error_threshold
        and result.max_error <= config.primitive_fit_error_threshold * 1.6
        and 0.84 <= result.area_ratio <= 1.16
        and result.confidence >= config.rectangle_confidence_threshold
    )


def _is_rounded_rectangle_candidate(result: RoundedRectangleFitResult | None, config: HybridVectorizerConfig) -> bool:
    if result is None:
        return False

    min_dimension = min(result.primitive.width, result.primitive.height)
    radius_ratio = result.primitive.corner_radius / max(1e-6, min_dimension)
    return (
        result.mean_error <= config.primitive_fit_error_threshold
        and result.max_error <= config.primitive_fit_error_threshold * 1.65
        and 0.8 <= result.area_ratio <= 1.2
        and 0.08 <= radius_ratio <= 0.28
        and result.confidence >= config.rounded_rectangle_confidence_threshold
    )


def fit_freeform_segments(
    points: list[Point],
    config: HybridVectorizerConfig,
    contour: ClosedContour | None = None,
) -> list[Segment]:
    if len(points) < 2:
        return []

    working_points = _prepare_freeform_points(points, config, contour=contour)
    return _fit_cubic_freeform_segments(working_points, config, fit_error=config.bezier_fit_tolerance)


def _prepare_freeform_points(
    points: list[Point],
    config: HybridVectorizerConfig,
    contour: ClosedContour | None,
) -> list[Point]:
    working = merge_near_duplicate_points(
        points,
        merge_distance=max(0.01, config.merge_distance * 0.35),
    )
    working = _simplify_closed_points_for_fit(working, config)
    if contour is not None and _is_shadow_like_contour(contour):
        working = _simplify_shadow_points(working, config)
    return working if len(working) >= 3 else points


def _fit_cubic_freeform_segments(
    points: list[Point],
    config: HybridVectorizerConfig,
    fit_error: float | None = None,
) -> list[Segment]:
    if len(points) < 2:
        return []

    curve_error = config.bezier_fit_tolerance if fit_error is None else float(fit_error)
    corner_candidates = classify_contour_corners(points, config)
    anchor_indices = _build_freeform_anchor_indices(points, corner_candidates)
    anchor_indices = _prune_freeform_anchor_indices(points, anchor_indices, corner_candidates)
    if len(anchor_indices) < 2:
        return _fit_closed_cubic_curve(points, error=curve_error)
    if len(anchor_indices) > max(10, len(points) // 7):
        return _fit_closed_cubic_curve(points, error=curve_error)

    segments: list[Segment] = []
    for span in split_closed_contour_at_corners(points, anchor_indices):
        span = merge_near_duplicate_points(
            span,
            merge_distance=max(0.01, config.merge_distance * 0.3),
            closed=False,
        )
        if len(span) < 2:
            continue
        segments.extend(_fit_span_between_corners(span, error=curve_error))

    if segments:
        return segments
    return _fit_closed_cubic_curve(points, error=curve_error)


def _build_freeform_anchor_indices(
    points: list[Point],
    corner_candidates: dict[int, CornerCandidate],
) -> list[int]:
    count = len(points)
    if count < 2:
        return []

    hard_corner_indices = detect_hard_corner_indices(
        points,
        angle_threshold_deg=HARD_CORNER_MIN_TURN_DEGREES,
    )
    anchors = sorted(
        index
        for index in range(count)
        if (
            (corner_candidates.get(index) is not None and corner_candidates[index].classification == "preserve_sharp")
            or index in hard_corner_indices
        )
    )
    if not anchors:
        return []
    if len(anchors) == 1:
        return sorted({anchors[0], (anchors[0] + count // 2) % count})
    return anchors


def _prune_freeform_anchor_indices(
    points: list[Point],
    anchors: list[int],
    corner_candidates: dict[int, CornerCandidate],
) -> list[int]:
    count = len(points)
    if count < 8 or len(anchors) <= 2:
        return anchors

    min_separation = max(4, count // 36)
    kept: list[int] = []
    scores = {index: _anchor_strength(index, points, corner_candidates) for index in anchors}

    for anchor in sorted(anchors):
        if not kept:
            kept.append(anchor)
            continue

        prev = kept[-1]
        if _cyclic_index_distance(anchor, prev, count) >= min_separation:
            kept.append(anchor)
            continue

        if scores[anchor] > scores[prev]:
            kept[-1] = anchor

    if len(kept) > 1 and _cyclic_index_distance(kept[0], kept[-1], count) < min_separation:
        if scores[kept[-1]] > scores[kept[0]]:
            kept = kept[1:]
        else:
            kept = kept[:-1]

    max_anchor_count = max(8, min(14, count // 10))
    if len(kept) <= max_anchor_count:
        return kept

    strongest = sorted(kept, key=lambda index: scores[index], reverse=True)
    selected: list[int] = []
    for anchor in strongest:
        if all(_cyclic_index_distance(anchor, existing, count) >= min_separation for existing in selected):
            selected.append(anchor)
        if len(selected) >= max_anchor_count:
            break
    return sorted(selected)


def _anchor_strength(
    index: int,
    points: list[Point],
    corner_candidates: dict[int, CornerCandidate],
) -> float:
    candidate = corner_candidates.get(index)
    if candidate is not None:
        return max(0.0, 180.0 - candidate.angle_degrees)

    prev_point = points[index - 1]
    point = points[index]
    next_point = points[(index + 1) % len(points)]
    return max(0.0, 180.0 - _corner_angle_degrees(prev_point, point, next_point))


def _cyclic_index_distance(left: int, right: int, count: int) -> int:
    direct = abs(left - right)
    return min(direct, count - direct)


def _fit_profiled_open_span(
    points: list[Point],
    error: float,
    rdp_tolerance: float,
    min_segments: int,
    max_segments: int | None = None,
    sigma: float = 1.0,
) -> list[Segment]:
    prepared = _prepare_profiled_open_span(points, sigma=sigma, rdp_tolerance=rdp_tolerance)
    if len(prepared) < 2:
        return []

    segments = _fit_cubic_span(prepared, error=error, rdp_tolerance=rdp_tolerance)
    if len(segments) < min_segments and len(prepared) >= 6:
        split_index = _best_open_span_split_index(prepared)
        if 1 < split_index < len(prepared) - 2:
            refined_error = max(FITCURVES_REFINED_ERROR, error * 0.82)
            refined_tolerance = max(0.45, rdp_tolerance * 0.85)
            left = _fit_cubic_span(prepared[: split_index + 1], error=refined_error, rdp_tolerance=refined_tolerance)
            right = _fit_cubic_span(prepared[split_index:], error=refined_error, rdp_tolerance=refined_tolerance)
            if left and right:
                segments = left + right

    if max_segments is not None and len(segments) > max_segments:
        looser = _fit_cubic_span(prepared, error=max(error * 1.12, error + 0.2), rdp_tolerance=rdp_tolerance * 1.08)
        if looser and len(looser) < len(segments):
            segments = looser

    return segments


def _prepare_profiled_open_span(points: list[Point], sigma: float, rdp_tolerance: float) -> list[Point]:
    if len(points) <= 2:
        return points

    smoothed = _gaussian_smooth_open_points(points, sigma=sigma)
    reduced = _reduce_points_for_fit_curve(
        smoothed,
        closed=False,
        tolerance=rdp_tolerance,
    )
    return reduced if len(reduced) >= 2 else points


def _gaussian_smooth_open_points(points: list[Point], sigma: float) -> list[Point]:
    if len(points) < 5 or sigma <= 0.0:
        return points

    radius = max(1, int(np.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-((offsets * offsets) / max(1e-6, 2.0 * sigma * sigma)))
    kernel /= np.sum(kernel)

    smoothed = [points[0]]
    last_index = len(points) - 1
    for index in range(1, last_index):
        accum_x = 0.0
        accum_y = 0.0
        weight_total = 0.0
        for offset, weight in zip(range(-radius, radius + 1), kernel):
            sample_index = min(last_index, max(0, index + offset))
            sample = points[sample_index]
            accum_x += sample.x * float(weight)
            accum_y += sample.y * float(weight)
            weight_total += float(weight)
        smoothed.append(Point(accum_x / weight_total, accum_y / weight_total))
    smoothed.append(points[-1])
    return smoothed


def _best_open_span_split_index(points: list[Point]) -> int:
    if len(points) < 5:
        return len(points) // 2

    start = points[0]
    end = points[-1]
    best_index = len(points) // 2
    best_distance = -1.0
    for index in range(1, len(points) - 1):
        deviation = _point_to_segment_distance(points[index], start, end)
        if deviation > best_distance:
            best_distance = deviation
            best_index = index
    return best_index


def _mean_x(points: list[Point]) -> float:
    if not points:
        return 0.0
    return sum(point.x for point in points) / len(points)


def _center_weighted_extreme_index(points: list[Point], axis: str, center_x: float) -> int:
    if axis == "top":
        return min(range(len(points)), key=lambda idx: (points[idx].y, abs(points[idx].x - center_x)))
    if axis == "bottom":
        return max(range(len(points)), key=lambda idx: (points[idx].y, -abs(points[idx].x - center_x)))
    raise ValueError(f"Unsupported axis: {axis}")


def _detect_shadow_anchor_indices(points: list[Point], contour: ClosedContour) -> tuple[int, int, int] | None:
    bbox = contour.bbox
    if bbox.width <= 0.0 or bbox.height <= 0.0:
        return None

    center_x = bbox.min_x + bbox.width * 0.5
    top_rim_limit = bbox.min_y + bbox.height * 0.18
    notch_depth_limit = bbox.min_y + bbox.height * 0.78

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

    if not left_bridge_candidates or not right_bridge_candidates or not notch_candidates:
        return None

    left_bridge_index = max(left_bridge_candidates, key=lambda idx: points[idx].x)
    right_bridge_index = min(right_bridge_candidates, key=lambda idx: points[idx].x)
    notch_index = max(
        notch_candidates,
        key=lambda idx: (points[idx].y, -abs(points[idx].x - center_x)),
    )
    return left_bridge_index, notch_index, right_bridge_index


def _select_bottom_shadow_span(
    points: list[Point],
    right_bridge_index: int,
    left_bridge_index: int,
    notch_index: int,
) -> list[Point]:
    forward = _closed_span(points, right_bridge_index, left_bridge_index)
    reverse = list(reversed(_closed_span(points, left_bridge_index, right_bridge_index)))
    candidates = [candidate for candidate in [forward, reverse] if candidate]
    if not candidates:
        return []

    scored = []
    for candidate in candidates:
        mean_y = sum(point.y for point in candidate) / len(candidate)
        contains_notch = points[notch_index] in candidate[1:-1]
        scored.append((contains_notch, mean_y, candidate))
    scored.sort(key=lambda item: (item[0], -item[1]))
    return scored[0][2]


def _extrema_indices_for_points(points: list[Point]) -> set[int]:
    if not points:
        return set()

    center_x = sum(point.x for point in points) / len(points)
    center_y = sum(point.y for point in points) / len(points)
    top = min(range(len(points)), key=lambda idx: (points[idx].y, abs(points[idx].x - center_x)))
    bottom = max(range(len(points)), key=lambda idx: (points[idx].y, -abs(points[idx].x - center_x)))
    left = min(range(len(points)), key=lambda idx: (points[idx].x, abs(points[idx].y - center_y)))
    right = max(range(len(points)), key=lambda idx: (points[idx].x, -abs(points[idx].y - center_y)))
    return {top, bottom, left, right}


def _shadow_bridge_indices_from_points(points: list[Point]) -> set[int]:
    if len(points) < 6:
        return set()

    min_x = min(point.x for point in points)
    max_x = max(point.x for point in points)
    min_y = min(point.y for point in points)
    max_y = max(point.y for point in points)
    width = max_x - min_x
    height = max_y - min_y
    if width <= 0.0 or height <= 0.0:
        return set()

    center_x = min_x + width * 0.5
    top_rim_limit = min_y + height * 0.18
    notch_depth_limit = min_y + height * 0.78
    left_candidates = [idx for idx, point in enumerate(points) if point.x < center_x and point.y <= top_rim_limit]
    right_candidates = [idx for idx, point in enumerate(points) if point.x > center_x and point.y <= top_rim_limit]
    notch_candidates = [
        idx
        for idx, point in enumerate(points)
        if abs(point.x - center_x) <= width * 0.18 and point.y <= notch_depth_limit
    ]

    anchors: set[int] = set()
    if left_candidates:
        anchors.add(max(left_candidates, key=lambda idx: points[idx].x))
    if right_candidates:
        anchors.add(min(right_candidates, key=lambda idx: points[idx].x))
    if notch_candidates:
        anchors.add(max(notch_candidates, key=lambda idx: (points[idx].y, -abs(points[idx].x - center_x))))
    return anchors


def _fit_closed_cubic_curve(points: list[Point], error: float) -> list[Segment]:
    reduced_points = _reduce_points_for_fit_curve(
        points,
        closed=True,
        tolerance=FITCURVES_PRE_RDP_EPSILON,
    )
    closed_points = np.asarray([(point.x, point.y) for point in reduced_points], dtype=np.float64)
    if np.linalg.norm(closed_points[0] - closed_points[-1]) > 1e-6:
        closed_points = np.vstack([closed_points, closed_points[0]])
    base_error = max(FITCURVES_MIN_ERROR, max(1e-6, float(error)))
    segments = _segments_from_fit_curve(
        fit_curve(closed_points, error=base_error),
        closed=True,
    )
    if _needs_dense_refit(points, segments, closed=True):
        refined_points = _reduce_points_for_fit_curve(
            points,
            closed=True,
            tolerance=FITCURVES_REFINED_RDP_EPSILON,
        )
        refined_closed_points = np.asarray([(point.x, point.y) for point in refined_points], dtype=np.float64)
        if np.linalg.norm(refined_closed_points[0] - refined_closed_points[-1]) > 1e-6:
            refined_closed_points = np.vstack([refined_closed_points, refined_closed_points[0]])
        refined_segments = _segments_from_fit_curve(
            fit_curve(refined_closed_points, error=min(base_error, FITCURVES_REFINED_ERROR)),
            closed=True,
        )
        if len(refined_segments) > len(segments):
            return refined_segments
    return segments


def split_closed_contour_at_corners(points: list[Point], corner_indices: list[int] | set[int]) -> list[list[Point]]:
    if not points:
        return []

    ordered = sorted({index % len(points) for index in corner_indices})
    if len(ordered) < 2:
        return [points[:]]

    spans: list[list[Point]] = []
    for offset, start_index in enumerate(ordered):
        end_index = ordered[(offset + 1) % len(ordered)]
        spans.append(_closed_span(points, start_index, end_index))
    return spans


def _fit_span_between_corners(points: list[Point], error: float) -> list[Segment]:
    if len(points) <= 3 or _polyline_length(points, closed=False) <= SHORT_SHARP_SPAN_LENGTH:
        return [SegmentLine(start=points[0], end=points[-1])]
    return _fit_cubic_span(points, error=error)


def _fit_cubic_span(points: list[Point], error: float, rdp_tolerance: float = FITCURVES_PRE_RDP_EPSILON) -> list[Segment]:
    if len(points) < 2:
        return []
    if len(points) == 2:
        return [SegmentLine(start=points[0], end=points[1])]

    reduced_points = _reduce_points_for_fit_curve(
        points,
        closed=False,
        tolerance=rdp_tolerance,
    )
    coords = np.asarray([(point.x, point.y) for point in reduced_points], dtype=np.float64)
    base_error = max(FITCURVES_MIN_ERROR, max(1e-6, float(error)))
    fitted = fit_curve(coords, error=base_error)
    if not fitted:
        return [SegmentLine(start=points[0], end=points[-1])]

    segments = _segments_from_fit_curve(fitted, closed=False)
    if _needs_dense_refit(points, segments, closed=False):
        refined_points = _reduce_points_for_fit_curve(
            points,
            closed=False,
            tolerance=FITCURVES_REFINED_RDP_EPSILON,
        )
        refined_coords = np.asarray([(point.x, point.y) for point in refined_points], dtype=np.float64)
        refined_segments = _segments_from_fit_curve(
            fit_curve(refined_coords, error=min(base_error, FITCURVES_REFINED_ERROR)),
            closed=False,
        )
        if len(refined_segments) > len(segments):
            return refined_segments
    return segments


def _segments_from_fit_curve(
    cubic_segments: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    closed: bool,
) -> list[Segment]:
    cubic_segments = _merge_micro_cubic_segments(
        cubic_segments,
        min_length=FITCURVES_MIN_SEGMENT_LENGTH,
        closed=closed,
    )
    return [
        SegmentBezierCubic(
            start=Point(float(segment[0][0]), float(segment[0][1])),
            control1=Point(float(segment[1][0]), float(segment[1][1])),
            control2=Point(float(segment[2][0]), float(segment[2][1])),
            end=Point(float(segment[3][0]), float(segment[3][1])),
        )
        for segment in cubic_segments
    ]


def _is_shadow_like_contour(contour: ClosedContour) -> bool:
    if contour.is_hole:
        return False
    bbox = contour.bbox
    if bbox.width <= 0.0 or bbox.height <= 0.0:
        return False
    aspect_ratio = bbox.width / bbox.height
    area_ratio = contour.area / max(1e-6, bbox.width * bbox.height)
    return aspect_ratio >= 1.7 and 0.45 <= area_ratio <= 0.92


def _simplify_shadow_points(points: list[Point], config: HybridVectorizerConfig) -> list[Point]:
    corner_candidates = classify_contour_corners(points, config)
    hard_corner_indices = detect_hard_corner_indices(
        points,
        angle_threshold_deg=HARD_CORNER_MIN_TURN_DEGREES,
    )
    protected_indices = {
        index
        for index, candidate in corner_candidates.items()
        if candidate.classification == "preserve_sharp"
    }
    protected_indices |= hard_corner_indices
    protected_indices |= _shadow_bridge_indices_from_points(points)
    simplified = simplify_closed_preserving_indices(
        points,
        tolerance=max(0.9, min(1.2, FITCURVES_PRE_RDP_EPSILON * 1.35)),
        protected_indices=protected_indices,
    )
    simplified = merge_near_duplicate_points(
        simplified,
        merge_distance=max(0.01, config.merge_distance * 0.35),
    )
    if len(simplified) < 4:
        return points
    return simplified


def _simplify_closed_points_for_fit(points: list[Point], config: HybridVectorizerConfig) -> list[Point]:
    if len(points) <= 4:
        return points

    corner_candidates = classify_contour_corners(points, config)
    hard_corner_indices = detect_hard_corner_indices(
        points,
        angle_threshold_deg=HARD_CORNER_MIN_TURN_DEGREES,
    )
    protected_indices = {
        index
        for index, candidate in corner_candidates.items()
        if candidate.classification == "preserve_sharp"
    }
    protected_indices |= hard_corner_indices
    reduced = simplify_closed_preserving_indices(
        points,
        tolerance=FITCURVES_PRE_RDP_EPSILON,
        protected_indices=protected_indices,
    )
    reduced = merge_near_duplicate_points(
        reduced,
        merge_distance=max(0.01, config.merge_distance * 0.3),
    )
    return reduced if len(reduced) >= 4 else points


def _reduce_points_for_fit_curve(points: list[Point], closed: bool, tolerance: float) -> list[Point]:
    if len(points) <= 3:
        return points

    reduced = (
        douglas_peucker_closed(points, tolerance=tolerance)
        if closed
        else douglas_peucker_open(points, tolerance=tolerance)
    )
    reduced = merge_near_duplicate_points(
        reduced,
        merge_distance=max(0.01, tolerance * 0.15),
        closed=closed,
    )
    minimum_points = 4 if closed else 2
    return reduced if len(reduced) >= minimum_points else points


def _merge_micro_cubic_segments(
    cubic_segments: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    min_length: float,
    closed: bool,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    segments = [
        tuple(np.asarray(control, dtype=np.float64).copy() for control in segment)
        for segment in cubic_segments
    ]
    if len(segments) <= 1:
        return segments

    index = 0
    while len(segments) > 1 and index < len(segments):
        if _cubic_chord_length(segments[index]) >= min_length:
            index += 1
            continue

        prev_exists = closed or index > 0
        next_exists = closed or index < len(segments) - 1
        if not prev_exists and not next_exists:
            break

        prev_index = (index - 1) % len(segments)
        next_index = (index + 1) % len(segments)
        if prev_exists and next_exists:
            merge_with_prev = _cubic_chord_length(segments[prev_index]) <= _cubic_chord_length(segments[next_index])
        else:
            merge_with_prev = prev_exists

        if merge_with_prev:
            if closed and index == 0:
                merged = _merge_adjacent_cubic_segments(segments[-1], segments[0])
                segments = [merged, *segments[1:-1]]
            else:
                merged = _merge_adjacent_cubic_segments(segments[index - 1], segments[index])
                segments[index - 1 : index + 1] = [merged]
                index = max(0, index - 1)
        else:
            if closed and index == len(segments) - 1:
                merged = _merge_adjacent_cubic_segments(segments[index], segments[0])
                segments = [merged, *segments[1:index]]
                index = 0
            else:
                merged = _merge_adjacent_cubic_segments(segments[index], segments[index + 1])
                segments[index : index + 2] = [merged]
        index = max(0, min(index, len(segments) - 1))
    return segments


def _merge_adjacent_cubic_segments(
    left: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    right: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        left[0].copy(),
        left[1].copy(),
        right[2].copy(),
        right[3].copy(),
    )


def _cubic_chord_length(segment: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> float:
    return float(np.linalg.norm(segment[3] - segment[0]))


def _needs_dense_refit(points: list[Point], segments: list[Segment], closed: bool) -> bool:
    if len(points) < 8 or len(segments) >= 12:
        return False

    span_length = _polyline_length(points, closed=closed)
    if span_length < 120.0:
        return False

    divisor = 120.0 if closed else 140.0
    target_segments = max(3 if closed else 2, min(12, int(round(span_length / divisor))))
    return len(segments) < target_segments


def _polyline_length(points: list[Point], closed: bool) -> float:
    if len(points) < 2:
        return 0.0

    limit = len(points) if closed else len(points) - 1
    total = 0.0
    for index in range(limit):
        total += _distance(points[index], points[(index + 1) % len(points)])
    return total


def _fit_curve_segments(points: list[Point], config: HybridVectorizerConfig) -> list[Segment]:
    if len(points) < 2:
        return []

    corner_candidates = classify_contour_corners(points, config)
    anchors = _build_anchor_indices(points, corner_candidates)
    segments: list[Segment] = []

    for anchor_index, start_index in enumerate(anchors):
        end_index = anchors[(anchor_index + 1) % len(anchors)]
        start_corner = corner_candidates.get(start_index)
        end_corner = corner_candidates.get(end_index)
        span = _build_trimmed_span(points, start_index, end_index, start_corner, end_corner)
        if len(span) >= 2:
            segments.extend(_fit_open_span(span, config))

        fillet = _corner_fillet_segment(end_corner)
        if fillet is not None:
            segments.append(fillet)
    return _merge_adjacent_lines(segments)


def _build_anchor_indices(points: list[Point], corner_candidates: dict[int, CornerCandidate]) -> list[int]:
    count = len(points)
    if count < 2:
        return [0]

    corner_indices = sorted(
        index
        for index, candidate in corner_candidates.items()
        if candidate.classification != "treat_as_smooth_curve"
    )

    if not corner_indices:
        step = max(1, count // 4)
        anchors = sorted({0, step % count, (2 * step) % count, (3 * step) % count})
        return anchors if len(anchors) >= 2 else [0, count // 2]

    if len(corner_indices) == 1:
        return sorted({corner_indices[0], (corner_indices[0] + count // 2) % count})
    return corner_indices


def _closed_span(points: list[Point], start_index: int, end_index: int) -> list[Point]:
    if start_index <= end_index:
        return points[start_index : end_index + 1]
    return points[start_index:] + points[: end_index + 1]


def _build_trimmed_span(
    points: list[Point],
    start_index: int,
    end_index: int,
    start_corner: CornerCandidate | None,
    end_corner: CornerCandidate | None,
) -> list[Point]:
    span = _closed_span(points, start_index, end_index)
    if not span:
        return []

    start_point = _corner_exit_point(start_corner) or span[0]
    end_point = _corner_entry_point(end_corner) or span[-1]

    trimmed: list[Point] = [start_point]
    for point in span[1:-1]:
        if _distance(trimmed[-1], point) > 1e-6:
            trimmed.append(point)
    if _distance(trimmed[-1], end_point) > 1e-6:
        trimmed.append(end_point)
    return trimmed if len(trimmed) >= 2 else [start_point, end_point]


def _corner_entry_point(candidate: CornerCandidate | None) -> Point | None:
    if candidate is None or candidate.classification != "apply_small_fillet" or candidate.fillet is None:
        return None
    return candidate.fillet.entry_point


def _corner_exit_point(candidate: CornerCandidate | None) -> Point | None:
    if candidate is None or candidate.classification != "apply_small_fillet" or candidate.fillet is None:
        return None
    return candidate.fillet.exit_point


def _corner_fillet_segment(candidate: CornerCandidate | None) -> Segment | None:
    if candidate is None or candidate.classification != "apply_small_fillet" or candidate.fillet is None:
        return None
    return candidate.fillet.as_arc()


def _fit_open_span(points: list[Point], config: HybridVectorizerConfig) -> list[Segment]:
    if len(points) < 2 or _distance(points[0], points[-1]) <= 1e-6:
        return []

    points = _simplify_open_span(points, config)
    max_curve_error = min(0.5, config.bezier_fit_tolerance)
    if len(points) <= 2:
        return [_line_as_cubic(points[0], points[-1])]

    cubic, cubic_error, cubic_split = _fit_cubic_bezier(points)
    if cubic is not None and cubic_error <= max_curve_error:
        return [cubic]

    quadratic, quadratic_error, quadratic_split = _fit_quadratic_bezier(points)
    if quadratic is not None and quadratic_error <= max_curve_error:
        return [_quadratic_to_cubic(quadratic)]

    split_index = cubic_split if cubic_error <= quadratic_error else quadratic_split
    if split_index <= 1 or split_index >= len(points) - 1:
        return _fit_catmull_rom_chain(points)

    left = _fit_open_span(points[: split_index + 1], config)
    right = _fit_open_span(points[split_index:], config)
    return left + right


def _fit_quadratic_bezier(points: list[Point]) -> tuple[SegmentBezierQuadratic | None, float, int]:
    if len(points) < 3:
        return None, float("inf"), -1

    parameters = _centripetal_parameterize(points)
    best_bezier: SegmentBezierQuadratic | None = None
    best_error = float("inf")
    best_split = -1

    for _ in range(4):
        bezier = _solve_quadratic_bezier(points, parameters)
        if bezier is None:
            break

        max_error, split_index = _bezier_max_error(points, parameters, lambda t: _evaluate_quadratic_bezier(bezier, t))
        if max_error < best_error:
            best_bezier = bezier
            best_error = max_error
            best_split = split_index

        refined = _refine_parameters(points, parameters, lambda t: _evaluate_quadratic_bezier(bezier, t))
        if refined is None or _parameter_shift(refined, parameters) <= 1e-3:
            break
        parameters = refined

    return best_bezier, best_error, best_split


def _fit_cubic_bezier(points: list[Point]) -> tuple[SegmentBezierCubic | None, float, int]:
    if len(points) < 4:
        return None, float("inf"), -1

    parameters = _centripetal_parameterize(points)
    best_bezier: SegmentBezierCubic | None = None
    best_error = float("inf")
    best_split = -1

    for _ in range(4):
        bezier = _solve_cubic_bezier(points, parameters)
        if bezier is None:
            break

        max_error, split_index = _bezier_max_error(points, parameters, lambda t: _evaluate_cubic_bezier(bezier, t))
        if max_error < best_error:
            best_bezier = bezier
            best_error = max_error
            best_split = split_index

        refined = _refine_parameters(points, parameters, lambda t: _evaluate_cubic_bezier(bezier, t))
        if refined is None or _parameter_shift(refined, parameters) <= 1e-3:
            break
        parameters = refined

    return best_bezier, best_error, best_split


def _fit_catmull_rom_chain(points: list[Point]) -> list[Segment]:
    if len(points) <= 2:
        return [_line_as_cubic(points[0], points[-1])]

    segments: list[Segment] = []
    for index in range(len(points) - 1):
        p0 = points[index - 1] if index > 0 else points[index]
        p1 = points[index]
        p2 = points[index + 1]
        p3 = points[index + 2] if index + 2 < len(points) else points[index + 1]
        c1 = Point(p1.x + (p2.x - p0.x) / 6.0, p1.y + (p2.y - p0.y) / 6.0)
        c2 = Point(p2.x - (p3.x - p1.x) / 6.0, p2.y - (p3.y - p1.y) / 6.0)
        segments.append(SegmentBezierCubic(start=p1, control1=c1, control2=c2, end=p2))
    return segments


def _line_as_cubic(start: Point, end: Point) -> SegmentBezierCubic:
    dx = end.x - start.x
    dy = end.y - start.y
    return SegmentBezierCubic(
        start=start,
        control1=Point(start.x + dx / 3.0, start.y + dy / 3.0),
        control2=Point(start.x + (2.0 * dx) / 3.0, start.y + (2.0 * dy) / 3.0),
        end=end,
    )


def _quadratic_to_cubic(segment: SegmentBezierQuadratic) -> SegmentBezierCubic:
    return SegmentBezierCubic(
        start=segment.start,
        control1=Point(
            segment.start.x + (2.0 / 3.0) * (segment.control.x - segment.start.x),
            segment.start.y + (2.0 / 3.0) * (segment.control.y - segment.start.y),
        ),
        control2=Point(
            segment.end.x + (2.0 / 3.0) * (segment.control.x - segment.end.x),
            segment.end.y + (2.0 / 3.0) * (segment.control.y - segment.end.y),
        ),
        end=segment.end,
    )


def _fit_arc_span(points: list[Point], config: HybridVectorizerConfig) -> Segment | None:
    candidates: list[tuple[float, Segment]] = []

    circular_result = fit_circular_arc(points)
    if _is_circular_arc_candidate(circular_result, config):
        candidates.append((circular_result.confidence - circular_result.normalized_max_error, circular_result.segment))

    elliptical_result = fit_elliptical_arc(points)
    if _is_elliptical_arc_candidate(elliptical_result, config):
        candidates.append((elliptical_result.confidence - elliptical_result.normalized_max_error, elliptical_result.segment))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _is_circular_arc_candidate(result: CircularArcFitResult | None, config: HybridVectorizerConfig) -> bool:
    if result is None:
        return False
    return (
        result.normalized_max_error <= config.arc_fit_tolerance
        and result.sweep_degrees >= config.min_arc_sweep_degrees
        and result.confidence >= config.arc_confidence_threshold
    )


def _is_elliptical_arc_candidate(result: EllipticalArcFitResult | None, config: HybridVectorizerConfig) -> bool:
    if result is None:
        return False
    return (
        result.normalized_max_error <= max(config.arc_fit_tolerance, config.ellipse_fit_tolerance)
        and result.sweep_degrees >= config.min_arc_sweep_degrees
        and result.confidence >= config.arc_confidence_threshold
    )


def _line_fit_error(points: list[Point]) -> float:
    start = points[0]
    end = points[-1]
    return max(_point_to_segment_distance(point, start, end) for point in points[1:-1]) if len(points) > 2 else 0.0


def _point_to_segment_distance(point: Point, start: Point, end: Point) -> float:
    dx = end.x - start.x
    dy = end.y - start.y
    if dx == 0.0 and dy == 0.0:
        return _distance(point, start)

    t = ((point.x - start.x) * dx + (point.y - start.y) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    projection = Point(start.x + t * dx, start.y + t * dy)
    return _distance(point, projection)


def _chord_length_parameterize(points: list[Point]) -> list[float]:
    distances = [0.0]
    total = 0.0
    for index in range(1, len(points)):
        total += _distance(points[index - 1], points[index])
        distances.append(total)
    if total <= 0.0:
        return [0.0 for _ in points]
    return [value / total for value in distances]


def _centripetal_parameterize(points: list[Point]) -> list[float]:
    distances = [0.0]
    total = 0.0
    for index in range(1, len(points)):
        total += max(1e-6, _distance(points[index - 1], points[index]) ** 0.5)
        distances.append(total)
    if total <= 0.0:
        return _chord_length_parameterize(points)
    return [value / total for value in distances]


def _solve_quadratic_bezier(points: list[Point], parameters: list[float]) -> SegmentBezierQuadratic | None:
    p0 = points[0]
    p2 = points[-1]
    denom = 0.0
    accum_x = 0.0
    accum_y = 0.0
    for point, t in zip(points[1:-1], parameters[1:-1]):
        basis = 2.0 * (1.0 - t) * t
        if basis <= 1e-6:
            continue
        base_x = (1.0 - t) * (1.0 - t) * p0.x + t * t * p2.x
        base_y = (1.0 - t) * (1.0 - t) * p0.y + t * t * p2.y
        accum_x += basis * (point.x - base_x)
        accum_y += basis * (point.y - base_y)
        denom += basis * basis
    if denom <= 1e-8:
        return None
    return SegmentBezierQuadratic(start=p0, control=Point(accum_x / denom, accum_y / denom), end=p2)


def _solve_cubic_bezier(points: list[Point], parameters: list[float]) -> SegmentBezierCubic | None:
    p0 = points[0]
    p3 = points[-1]
    basis_rows = []
    rhs_x = []
    rhs_y = []
    for point, t in zip(points[1:-1], parameters[1:-1]):
        omt = 1.0 - t
        b0 = omt * omt * omt
        b1 = 3.0 * omt * omt * t
        b2 = 3.0 * omt * t * t
        b3 = t * t * t
        basis_rows.append([b1, b2])
        rhs_x.append(point.x - (b0 * p0.x + b3 * p3.x))
        rhs_y.append(point.y - (b0 * p0.y + b3 * p3.y))

    matrix = np.asarray(basis_rows, dtype=np.float64)
    if matrix.shape[0] < 2:
        return None
    try:
        ctrl_x, *_ = np.linalg.lstsq(matrix, np.asarray(rhs_x, dtype=np.float64), rcond=None)
        ctrl_y, *_ = np.linalg.lstsq(matrix, np.asarray(rhs_y, dtype=np.float64), rcond=None)
    except np.linalg.LinAlgError:
        return None

    if len(ctrl_x) < 2 or len(ctrl_y) < 2:
        return None

    return SegmentBezierCubic(
        start=p0,
        control1=Point(float(ctrl_x[0]), float(ctrl_y[0])),
        control2=Point(float(ctrl_x[1]), float(ctrl_y[1])),
        end=p3,
    )


def _bezier_max_error(
    points: list[Point],
    parameters: list[float],
    evaluator: Callable[[float], Point],
) -> tuple[float, int]:
    max_error = 0.0
    split_index = len(points) // 2
    for index, (point, t) in enumerate(zip(points, parameters)):
        sample = evaluator(t)
        error = _distance(point, sample)
        if error > max_error:
            max_error = error
            split_index = index
    return max_error, split_index


def _refine_parameters(
    points: list[Point],
    parameters: list[float],
    evaluator: Callable[[float], Point],
) -> list[float] | None:
    if len(points) <= 2:
        return parameters

    sample_count = max(32, len(points) * 6)
    sample_ts = np.linspace(0.0, 1.0, sample_count + 1, dtype=np.float64)
    samples = [evaluator(float(t)) for t in sample_ts]
    refined = [0.0]
    remaining = len(points) - 2

    for point in points[1:-1]:
        lower = refined[-1] + 1e-4
        upper = 1.0 - remaining * 1e-4
        best_t = None
        best_error = float("inf")
        for candidate_t, sample in zip(sample_ts, samples):
            value = float(candidate_t)
            if value < lower or value > upper:
                continue
            error = _distance(point, sample)
            if error < best_error:
                best_error = error
                best_t = value
        if best_t is None:
            return None
        refined.append(best_t)
        remaining -= 1

    refined.append(1.0)
    return refined if len(refined) == len(parameters) else None


def _parameter_shift(left: list[float], right: list[float]) -> float:
    return max(abs(a - b) for a, b in zip(left, right))


def _evaluate_quadratic_bezier(bezier: SegmentBezierQuadratic, t: float) -> Point:
    omt = 1.0 - t
    x = omt * omt * bezier.start.x + 2.0 * omt * t * bezier.control.x + t * t * bezier.end.x
    y = omt * omt * bezier.start.y + 2.0 * omt * t * bezier.control.y + t * t * bezier.end.y
    return Point(x, y)


def _evaluate_cubic_bezier(bezier: SegmentBezierCubic, t: float) -> Point:
    omt = 1.0 - t
    x = (
        omt * omt * omt * bezier.start.x
        + 3.0 * omt * omt * t * bezier.control1.x
        + 3.0 * omt * t * t * bezier.control2.x
        + t * t * t * bezier.end.x
    )
    y = (
        omt * omt * omt * bezier.start.y
        + 3.0 * omt * omt * t * bezier.control1.y
        + 3.0 * omt * t * t * bezier.control2.y
        + t * t * t * bezier.end.y
    )
    return Point(x, y)


def _merge_adjacent_lines(segments: list[Segment]) -> list[Segment]:
    if not segments:
        return segments

    merged: list[Segment] = [segments[0]]
    for segment in segments[1:]:
        prev = merged[-1]
        if isinstance(prev, SegmentLine) and isinstance(segment, SegmentLine) and _are_lines_collinear(prev, segment):
            merged[-1] = SegmentLine(start=prev.start, end=segment.end)
            continue
        merged.append(segment)
    return merged


def _are_lines_collinear(left: SegmentLine, right: SegmentLine) -> bool:
    if left.end != right.start:
        return False
    tolerance = 0.25
    return _point_to_segment_distance(left.end, left.start, right.end) <= tolerance


def _distance(a: Point, b: Point) -> float:
    return float(np.hypot(a.x - b.x, a.y - b.y))


def _simplify_open_span(points: list[Point], config: HybridVectorizerConfig) -> list[Point]:
    if len(points) <= 3:
        return points

    tolerance = max(0.18, config.simplify_tolerance * 0.35)
    reduced = douglas_peucker_open(points, tolerance=tolerance)
    reduced = merge_near_duplicate_points(reduced, merge_distance=max(0.01, config.merge_distance * 0.45), closed=False)
    return reduced if len(reduced) >= 2 else points


def _fit_safe_spline_loop(contour: ClosedContour, config: HybridVectorizerConfig, polarity: str) -> Loop | None:
    reduced = douglas_peucker_closed(contour.points, tolerance=max(0.4, config.simplify_tolerance * 0.55))
    reduced = merge_near_duplicate_points(reduced, merge_distance=max(0.01, config.merge_distance * 0.5))
    if len(reduced) < 4:
        return None

    corner_candidates = classify_contour_corners(reduced, config)
    sharp_corners = {
        index
        for index, candidate in corner_candidates.items()
        if candidate.classification == "preserve_sharp"
    }

    segments: list[Segment] = []
    count = len(reduced)
    tension = 0.16
    for index in range(count):
        start = reduced[index]
        end = reduced[(index + 1) % count]
        if index in sharp_corners or (index + 1) % count in sharp_corners:
            segments.append(_line_as_cubic(start, end))
            continue

        prev_point = reduced[index - 1]
        next_point = reduced[(index + 2) % count]
        control1 = Point(
            start.x + (end.x - prev_point.x) * tension,
            start.y + (end.y - prev_point.y) * tension,
        )
        control2 = Point(
            end.x - (next_point.x - start.x) * tension,
            end.y - (next_point.y - start.y) * tension,
        )
        segments.append(
            SegmentBezierCubic(
                start=start,
                control1=control1,
                control2=control2,
                end=end,
            )
        )

    return Loop(
        loop_id=contour.contour_id,
        segments=_merge_adjacent_lines(segments),
        polarity=polarity,
        closed=True,
        source_contour_id=contour.contour_id,
        confidence=0.0,
    )


def _polyline_loop_from_contour(contour: ClosedContour, polarity: str) -> Loop:
    return build_polyline_loop(
        loop_id=contour.contour_id,
        points=contour.points,
        polarity=polarity,
        source_contour_id=contour.contour_id,
        confidence=0.0,
    )
