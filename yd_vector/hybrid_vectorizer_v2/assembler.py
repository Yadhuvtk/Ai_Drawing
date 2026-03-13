from __future__ import annotations

from yd_vector.hybrid_vectorizer.geometry import Shape
from yd_vector.hybrid_vectorizer.loop_builder import build_polyline_loop
from yd_vector.hybrid_vectorizer_v2.config import HybridVectorizerV2Config
from yd_vector.hybrid_vectorizer_v2.decompose import decompose_region
from yd_vector.hybrid_vectorizer_v2.freeform import fit_freeform_loop
from yd_vector.hybrid_vectorizer_v2.geometry import ContourPartPlan, RegionDecomposition
from yd_vector.hybrid_vectorizer_v2.topology import validate_loop, validate_shape


def assemble_region_shape(
    decomposition: RegionDecomposition,
    config: HybridVectorizerV2Config,
    shape_id: str | None = None,
) -> Shape:
    outer_loop = _select_loop(decomposition.outer, config)
    negative_loops = [_select_loop(hole, config) for hole in decomposition.holes]
    shape = Shape(
        shape_id=shape_id or decomposition.region.region_id,
        outer_loop=outer_loop,
        negative_loops=negative_loops,
        fill=config.fill_color,
        stroke=config.stroke_color,
    )
    if validate_shape(shape, decomposition.region.outer, decomposition.region.holes, config):
        return shape

    return Shape(
        shape_id=shape_id or decomposition.region.region_id,
        outer_loop=_polyline_loop(decomposition.outer),
        negative_loops=[_polyline_loop(hole) for hole in decomposition.holes],
        fill=config.fill_color,
        stroke=config.stroke_color,
    )


def decompose_and_assemble(region: object, config: HybridVectorizerV2Config) -> Shape:
    return assemble_region_shape(decompose_region(region, config), config)


def _select_loop(plan: ContourPartPlan, config: HybridVectorizerV2Config):
    for candidate in plan.primitive_candidates:
        if validate_loop(candidate.loop, plan.contour, config):
            return candidate.loop

    freeform_loop = fit_freeform_loop(plan, config)
    if validate_loop(freeform_loop, plan.contour, config):
        return freeform_loop
    return _polyline_loop(plan)


def _polyline_loop(plan: ContourPartPlan):
    return build_polyline_loop(
        loop_id=plan.contour.contour_id,
        points=plan.contour.points,
        polarity=plan.polarity,
        source_contour_id=plan.contour.contour_id,
        confidence=0.0,
    )
