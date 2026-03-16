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
    bilateral_filter_diameter: int = 5
    bilateral_sigma_color: float = 22.0
    bilateral_sigma_space: float = 2.2
    preblur_radius: float = 0.9
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
    simplify_tolerance: float = 0.8
    merge_distance: float = 1.0
    corner_angle_threshold_degrees: float = 115.0
    smooth_iterations: int = 3
    smooth_strength: float = 0.42
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
    primitive_fit_error_threshold: float = 1.0
    topology_area_tolerance: float = 0.12
    hole_area_tolerance: float = 0.16
    topology_bbox_iou_threshold: float = 0.68
    hole_bbox_iou_threshold: float = 0.56
    circle_fit_tolerance: float = 0.055
    circle_circularity_min: float = 0.84
    circle_confidence_threshold: float = 0.9
    ellipse_fit_tolerance: float = 0.085
    ellipse_confidence_threshold: float = 0.88
    rectangle_confidence_threshold: float = 0.92
    rounded_rectangle_confidence_threshold: float = 0.9
    line_fit_tolerance: float = 1.0
    arc_fit_tolerance: float = 0.08
    arc_confidence_threshold: float = 0.78
    min_arc_sweep_degrees: float = 18.0
    quadratic_fit_tolerance: float = 0.5
    bezier_fit_tolerance: float = 2.0
    fill_color: str = "#000000"
    stroke_color: str | None = None
    background: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "HybridVectorizerConfig":
        fields = cls.__dataclass_fields__.keys()
        return cls(**{key: data[key] for key in fields if key in data})
