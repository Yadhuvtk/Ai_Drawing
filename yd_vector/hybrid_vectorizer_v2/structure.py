from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from yd_vector.hybrid_vectorizer.cleanup import cleanup_region
from yd_vector.hybrid_vectorizer.config import HybridVectorizerConfig
from yd_vector.hybrid_vectorizer.contour_extraction import extract_regions
from yd_vector.hybrid_vectorizer.geometry import ContourRegion
from yd_vector.hybrid_vectorizer.preprocessing import PreprocessResult, preprocess_image
from yd_vector.hybrid_vectorizer_v2.config import HybridVectorizerV2Config


@dataclass
class ExtractedStructure:
    input_path: Path
    preprocessed: PreprocessResult
    regions: list[ContourRegion]


def extract_monochrome_structure(
    image_path: str | Path,
    config: HybridVectorizerV2Config,
) -> ExtractedStructure:
    input_path = Path(image_path)
    base_config = _to_base_config(config)
    preprocessed = preprocess_image(input_path, base_config)
    raw_regions = extract_regions(
        preprocessed.binary_mask,
        min_region_area=config.min_region_area,
        min_hole_area=config.min_hole_area,
        scalar_field=preprocessed.foreground,
        contour_level=preprocessed.foreground_threshold,
        subpixel=config.subpixel_contours,
    )
    regions = [cleanup_region(region, base_config) for region in raw_regions]
    return ExtractedStructure(
        input_path=input_path,
        preprocessed=preprocessed,
        regions=regions,
    )


def _to_base_config(config: HybridVectorizerV2Config) -> HybridVectorizerConfig:
    return HybridVectorizerConfig(
        vectorization_mode="monochrome",
        target_size=config.target_size,
        threshold=config.threshold,
        threshold_bias=config.threshold_bias,
        invert=config.invert,
        use_alpha_foreground=config.use_alpha_foreground,
        preblur_radius=config.preblur_radius,
        morph_open_iterations=config.morph_open_iterations,
        morph_close_iterations=config.morph_close_iterations,
        min_region_area=config.min_region_area,
        min_hole_area=config.min_hole_area,
        simplify_tolerance=config.simplify_tolerance,
        merge_distance=config.merge_distance,
        corner_angle_threshold_degrees=config.corner_angle_threshold_degrees,
        smooth_iterations=config.smooth_iterations,
        smooth_strength=config.smooth_strength,
        min_corner_angle_deg=config.min_corner_angle_deg,
        sharp_corner_angle_deg=config.sharp_corner_angle_deg,
        smooth_curve_angle_threshold=config.smooth_curve_angle_threshold,
        preserve_tip_points=True,
        preserve_narrow_gaps=config.preserve_narrow_gaps,
        gap_preservation_distance=config.gap_preservation_distance,
        gap_protection_span=config.gap_protection_span,
        subpixel_contours=config.subpixel_contours,
        primitive_fit_error_threshold=config.primitive_fit_error_threshold,
        topology_area_tolerance=config.topology_area_tolerance,
        hole_area_tolerance=config.hole_area_tolerance,
        topology_bbox_iou_threshold=config.topology_bbox_iou_threshold,
        hole_bbox_iou_threshold=config.hole_bbox_iou_threshold,
        circle_fit_tolerance=config.circle_fit_tolerance,
        circle_circularity_min=config.circle_circularity_min,
        circle_confidence_threshold=config.circle_confidence_threshold,
        ellipse_fit_tolerance=config.ellipse_fit_tolerance,
        ellipse_confidence_threshold=config.ellipse_confidence_threshold,
        rectangle_confidence_threshold=config.rectangle_confidence_threshold,
        rounded_rectangle_confidence_threshold=config.rounded_rectangle_confidence_threshold,
        line_fit_tolerance=config.line_fit_tolerance,
        arc_fit_tolerance=config.arc_fit_tolerance,
        arc_confidence_threshold=config.arc_confidence_threshold,
        min_arc_sweep_degrees=config.min_arc_sweep_degrees,
        quadratic_fit_tolerance=config.quadratic_fit_tolerance,
        bezier_fit_tolerance=config.bezier_fit_tolerance,
        fill_color=config.fill_color,
        stroke_color=config.stroke_color,
        background=config.background,
    )
