from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yd_vector.hybrid_vectorizer.evaluation import document_stats
from yd_vector.hybrid_vectorizer_v2 import HybridVectorizerV2Config, PrimitiveFirstVectorizerV2
from yd_vector.paths import REPO_ROOT


def _default_output_path(input_path: Path) -> Path:
    return REPO_ROOT / "outputs" / "hybrid_vectorizer_v2" / f"{input_path.stem}.svg"


def main() -> None:
    parser = argparse.ArgumentParser(description="Primitive-first one-color vectorizer v2.")
    parser.add_argument("--input", type=str, required=True, help="Path to the raster input image.")
    parser.add_argument("--output", type=str, default=None, help="Path to write the SVG output.")
    parser.add_argument("--target_size", type=int, default=None, help="Resize the longest side before extraction.")
    parser.add_argument("--threshold", type=int, default=None, help="Binary threshold. Defaults to Otsu.")
    parser.add_argument("--threshold_bias", type=int, default=0, help="Bias applied after threshold estimation.")
    parser.add_argument("--invert", action="store_true", help="Treat light foreground on dark background as foreground.")
    parser.add_argument("--disable_alpha_foreground", action="store_true", help="Ignore alpha when building the mask.")
    parser.add_argument("--preblur_radius", type=float, default=1.0, help="Blur radius before thresholding.")
    parser.add_argument("--morph_open_iterations", type=int, default=0, help="Optional opening passes.")
    parser.add_argument("--morph_close_iterations", type=int, default=0, help="Optional closing passes.")
    parser.add_argument("--min_region_area", type=int, default=32, help="Drop connected regions smaller than this.")
    parser.add_argument("--min_hole_area", type=int, default=16, help="Drop hole regions smaller than this.")
    parser.add_argument("--simplify_tolerance", type=float, default=1.0, help="Anchor simplification tolerance.")
    parser.add_argument("--merge_distance", type=float, default=1.0, help="Merge contour points closer than this.")
    parser.add_argument("--disable_gap_preservation", action="store_true", help="Allow narrow gaps to collapse during cleanup.")
    parser.add_argument("--gap_preservation_distance", type=float, default=10.0, help="Protect contour points across narrow gaps.")
    parser.add_argument("--gap_protection_span", type=int, default=2, help="Protected neighbor count around a narrow gap.")
    parser.add_argument("--disable_subpixel_contours", action="store_true", help="Use pixel-edge contours instead of subpixel loops.")
    parser.add_argument("--sharp_corner_angle_deg", type=float, default=92.0, help="Angles below this stay sharp in the freeform fallback.")
    parser.add_argument("--smooth_curve_angle_threshold", type=float, default=148.0, help="Angles above this are treated as smooth continuation.")
    parser.add_argument("--line_fit_tolerance", type=float, default=0.9, help="Maximum deviation for straight-run detection.")
    parser.add_argument("--arc_fit_tolerance", type=float, default=0.08, help="Maximum normalized arc-fit error.")
    parser.add_argument("--arc_confidence_threshold", type=float, default=0.80, help="Minimum confidence for arc promotion.")
    parser.add_argument("--primitive_fit_error_threshold", type=float, default=1.2, help="Maximum error allowed for rectangle-like primitives.")
    parser.add_argument("--circle_fit_tolerance", type=float, default=0.08, help="Tolerance for circle promotion.")
    parser.add_argument("--circle_confidence_threshold", type=float, default=0.88, help="Minimum confidence required for circles.")
    parser.add_argument("--ellipse_fit_tolerance", type=float, default=0.11, help="Tolerance for ellipse promotion.")
    parser.add_argument("--ellipse_confidence_threshold", type=float, default=0.84, help="Minimum confidence required for ellipses.")
    parser.add_argument("--min_ellipse_subshape_area", type=float, default=350.0, help="Minimum contour area before trying bottom ellipse subshape fitting.")
    parser.add_argument("--ellipse_subshape_min_aspect_ratio", type=float, default=1.2, help="Minimum aspect ratio for an ellipse-like base subshape.")
    parser.add_argument("--ellipse_subshape_max_aspect_ratio", type=float, default=4.5, help="Maximum aspect ratio for an ellipse-like base subshape.")
    parser.add_argument("--ellipse_subshape_band_ratio", type=float, default=0.58, help="Only points in the lower band above this ratio are considered for bottom ellipse fitting.")
    parser.add_argument("--ellipse_subshape_min_span_ratio", type=float, default=0.42, help="Minimum fraction of contour width that a bottom ellipse span must cover.")
    parser.add_argument("--ellipse_subshape_symmetry_tolerance", type=float, default=0.20, help="How much left-right ellipse asymmetry is tolerated before fallback.")
    parser.add_argument("--ellipse_subshape_confidence_threshold", type=float, default=0.62, help="Minimum confidence required before a lower ellipse-like span replaces the traced base contour.")
    parser.add_argument("--rectangle_confidence_threshold", type=float, default=0.90, help="Minimum confidence required for rectangles.")
    parser.add_argument("--rounded_rectangle_confidence_threshold", type=float, default=0.88, help="Minimum confidence required for rounded rectangles.")
    parser.add_argument("--topology_area_tolerance", type=float, default=0.16, help="Maximum relative area distortion before fallback.")
    parser.add_argument("--hole_area_tolerance", type=float, default=0.18, help="Maximum relative area distortion allowed for holes.")
    parser.add_argument("--topology_bbox_iou_threshold", type=float, default=0.60, help="Minimum bbox IoU required for outer loops.")
    parser.add_argument("--hole_bbox_iou_threshold", type=float, default=0.50, help="Minimum bbox IoU required for holes.")
    parser.add_argument("--fill", type=str, default="#000000", help="SVG fill color.")
    parser.add_argument("--stroke", type=str, default=None, help="Optional SVG stroke color.")
    parser.add_argument("--background", type=str, default=None, help="Optional SVG background.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = REPO_ROOT / input_path
    if not input_path.exists():
        raise FileNotFoundError(f"input image not found: {input_path}")

    output_path = Path(args.output) if args.output else _default_output_path(input_path)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path

    config = HybridVectorizerV2Config(
        target_size=args.target_size,
        threshold=args.threshold,
        threshold_bias=args.threshold_bias,
        invert=bool(args.invert),
        use_alpha_foreground=not args.disable_alpha_foreground,
        preblur_radius=args.preblur_radius,
        morph_open_iterations=args.morph_open_iterations,
        morph_close_iterations=args.morph_close_iterations,
        min_region_area=args.min_region_area,
        min_hole_area=args.min_hole_area,
        simplify_tolerance=args.simplify_tolerance,
        merge_distance=args.merge_distance,
        preserve_narrow_gaps=not args.disable_gap_preservation,
        gap_preservation_distance=args.gap_preservation_distance,
        gap_protection_span=args.gap_protection_span,
        subpixel_contours=not args.disable_subpixel_contours,
        sharp_corner_angle_deg=args.sharp_corner_angle_deg,
        smooth_curve_angle_threshold=args.smooth_curve_angle_threshold,
        line_fit_tolerance=args.line_fit_tolerance,
        arc_fit_tolerance=args.arc_fit_tolerance,
        arc_confidence_threshold=args.arc_confidence_threshold,
        primitive_fit_error_threshold=args.primitive_fit_error_threshold,
        circle_fit_tolerance=args.circle_fit_tolerance,
        circle_confidence_threshold=args.circle_confidence_threshold,
        ellipse_fit_tolerance=args.ellipse_fit_tolerance,
        ellipse_confidence_threshold=args.ellipse_confidence_threshold,
        min_ellipse_subshape_area=args.min_ellipse_subshape_area,
        ellipse_subshape_min_aspect_ratio=args.ellipse_subshape_min_aspect_ratio,
        ellipse_subshape_max_aspect_ratio=args.ellipse_subshape_max_aspect_ratio,
        ellipse_subshape_band_ratio=args.ellipse_subshape_band_ratio,
        ellipse_subshape_min_span_ratio=args.ellipse_subshape_min_span_ratio,
        ellipse_subshape_symmetry_tolerance=args.ellipse_subshape_symmetry_tolerance,
        ellipse_subshape_confidence_threshold=args.ellipse_subshape_confidence_threshold,
        rectangle_confidence_threshold=args.rectangle_confidence_threshold,
        rounded_rectangle_confidence_threshold=args.rounded_rectangle_confidence_threshold,
        topology_area_tolerance=args.topology_area_tolerance,
        hole_area_tolerance=args.hole_area_tolerance,
        topology_bbox_iou_threshold=args.topology_bbox_iou_threshold,
        hole_bbox_iou_threshold=args.hole_bbox_iou_threshold,
        fill_color=args.fill,
        stroke_color=args.stroke,
        background=args.background,
    )

    pipeline = PrimitiveFirstVectorizerV2(config)
    result = pipeline.vectorize(input_path, output_path=output_path)
    stats = document_stats(result.document)

    print(f"Input image: {input_path}")
    print(f"Output SVG: {output_path}")
    print(f"Threshold used: {result.structure.preprocessed.threshold}")
    print(f"Regions: {len(result.structure.regions)}")
    print(f"Shapes: {stats['num_shapes']}")
    print(f"Holes: {stats['num_holes']}")
    print(f"Loops: {stats['num_loops']}")
    print(f"Segments: {stats['num_segments']}")


if __name__ == "__main__":
    main()
