from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HybridVectorizerV2Config:
    target_size: int | None = None
    threshold: int | None = None
    threshold_bias: int = 0
    invert: bool = False
    use_alpha_foreground: bool = True
    preblur_radius: float = 1.0
    morph_open_iterations: int = 0
    morph_close_iterations: int = 0
    min_region_area: int = 32
    min_hole_area: int = 16
    simplify_tolerance: float = 1.0
    merge_distance: float = 1.0
    preserve_narrow_gaps: bool = True
    gap_preservation_distance: float = 10.0
    gap_protection_span: int = 2
    subpixel_contours: bool = True
    corner_angle_threshold_degrees: float = 118.0
    smooth_iterations: int = 1
    smooth_strength: float = 0.18
    min_corner_angle_deg: float = 124.0
    sharp_corner_angle_deg: float = 92.0
    smooth_curve_angle_threshold: float = 148.0
    line_fit_tolerance: float = 0.9
    arc_fit_tolerance: float = 0.08
    arc_confidence_threshold: float = 0.80
    min_arc_sweep_degrees: float = 18.0
    quadratic_fit_tolerance: float = 1.2
    bezier_fit_tolerance: float = 1.5
    cubic_tension: float = 0.72
    primitive_fit_error_threshold: float = 1.2
    circle_fit_tolerance: float = 0.08
    circle_circularity_min: float = 0.80
    circle_confidence_threshold: float = 0.88
    ellipse_fit_tolerance: float = 0.11
    ellipse_confidence_threshold: float = 0.84
    min_ellipse_subshape_area: float = 350.0
    ellipse_subshape_min_aspect_ratio: float = 1.2
    ellipse_subshape_max_aspect_ratio: float = 4.5
    ellipse_subshape_band_ratio: float = 0.58
    ellipse_subshape_min_span_ratio: float = 0.42
    ellipse_subshape_symmetry_tolerance: float = 0.20
    ellipse_subshape_rotation_tolerance_deg: float = 18.0
    ellipse_subshape_confidence_threshold: float = 0.62
    rectangle_confidence_threshold: float = 0.90
    rounded_rectangle_confidence_threshold: float = 0.88
    prefer_hole_primitives: bool = True
    topology_area_tolerance: float = 0.16
    hole_area_tolerance: float = 0.18
    topology_bbox_iou_threshold: float = 0.60
    hole_bbox_iou_threshold: float = 0.50
    fill_color: str = "#000000"
    stroke_color: str | None = None
    background: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "HybridVectorizerV2Config":
        fields = cls.__dataclass_fields__.keys()
        return cls(**{key: data[key] for key in fields if key in data})
