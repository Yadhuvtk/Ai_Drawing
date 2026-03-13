from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HybridVectorizerConfig:
    vectorization_mode: str = "monochrome"
    target_size: int | None = None
    threshold: int | None = None
    threshold_bias: int = 0
    invert: bool = False
    use_alpha_foreground: bool = True
    preblur_radius: float = 1.0
    color_preblur_radius: float = 0.0
    morph_open_iterations: int = 0
    morph_close_iterations: int = 0
    max_colors: int = 8
    color_distance_threshold: float = 28.0
    quantization_method: str = "median_cut"
    merge_similar_colors: bool = True
    background_mode: str = "auto"
    background_color: str | None = None
    min_region_area: int = 32
    min_hole_area: int = 16
    simplify_tolerance: float = 1.25
    merge_distance: float = 1.0
    corner_angle_threshold_degrees: float = 115.0
    smooth_iterations: int = 2
    smooth_strength: float = 0.35
    min_corner_angle_deg: float = 124.0
    sharp_corner_angle_deg: float = 92.0
    smooth_curve_angle_threshold: float = 144.0
    fillet_radius_ratio: float = 0.11
    max_fillet_radius: float = 3.25
    preserve_tip_points: bool = True
    preserve_narrow_gaps: bool = True
    gap_preservation_distance: float = 10.0
    gap_protection_span: int = 2
    subpixel_contours: bool = True
    primitive_fit_error_threshold: float = 1.35
    topology_area_tolerance: float = 0.18
    hole_area_tolerance: float = 0.22
    topology_bbox_iou_threshold: float = 0.55
    hole_bbox_iou_threshold: float = 0.42
    circle_fit_tolerance: float = 0.08
    circle_circularity_min: float = 0.78
    circle_confidence_threshold: float = 0.86
    ellipse_fit_tolerance: float = 0.12
    ellipse_confidence_threshold: float = 0.82
    rectangle_confidence_threshold: float = 0.88
    rounded_rectangle_confidence_threshold: float = 0.86
    line_fit_tolerance: float = 1.0
    arc_fit_tolerance: float = 0.08
    arc_confidence_threshold: float = 0.78
    min_arc_sweep_degrees: float = 18.0
    quadratic_fit_tolerance: float = 1.35
    bezier_fit_tolerance: float = 1.75
    fill_color: str = "#000000"
    stroke_color: str | None = None
    background: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "HybridVectorizerConfig":
        fields = cls.__dataclass_fields__.keys()
        return cls(**{key: data[key] for key in fields if key in data})
