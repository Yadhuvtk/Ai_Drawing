from __future__ import annotations

from yd_vector.hybrid_vectorizer.config import HybridVectorizerConfig
from yd_vector.hybrid_vectorizer.geometry import ClosedContour, Loop, Shape
from yd_vector.hybrid_vectorizer.topology_guard import sample_loop_points, validate_loop_against_contour, validate_shape_topology
from yd_vector.hybrid_vectorizer_v2.config import HybridVectorizerV2Config


def validate_loop(loop: Loop, contour: ClosedContour, config: HybridVectorizerV2Config) -> bool:
    return validate_loop_against_contour(loop, contour, _to_guard_config(config))


def validate_shape(shape: Shape, outer: ClosedContour, holes: list[ClosedContour], config: HybridVectorizerV2Config) -> bool:
    return validate_shape_topology(shape, outer, holes, _to_guard_config(config))


def sample_points(loop: Loop) -> list[object]:
    return sample_loop_points(loop)


def _to_guard_config(config: HybridVectorizerV2Config) -> HybridVectorizerConfig:
    return HybridVectorizerConfig(
        vectorization_mode="monochrome",
        topology_area_tolerance=config.topology_area_tolerance,
        hole_area_tolerance=config.hole_area_tolerance,
        topology_bbox_iou_threshold=config.topology_bbox_iou_threshold,
        hole_bbox_iou_threshold=config.hole_bbox_iou_threshold,
    )
