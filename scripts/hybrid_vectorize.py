from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yd_vector.hybrid_vectorizer import HybridVectorizerConfig, HybridVectorizerPipeline
from yd_vector.hybrid_vectorizer.evaluation import document_stats
from yd_vector.paths import REPO_ROOT


def _default_output_path(input_path: Path) -> Path:
    return REPO_ROOT / "outputs" / "hybrid_vectorizer" / f"{input_path.stem}.svg"


def main() -> None:
    parser = argparse.ArgumentParser(description="Vectorize a flat-color raster image with the hybrid vectorizer.")
    parser.add_argument("--input", type=str, required=True, help="Path to the raster input image.")
    parser.add_argument("--output", type=str, default=None, help="Path to write the SVG output.")
    parser.add_argument("--mode", type=str, default="monochrome", help="Vectorization mode: monochrome, auto, or flat_color.")
    parser.add_argument("--target_size", type=int, default=None, help="Resize the longest side before extraction.")
    parser.add_argument("--threshold", type=int, default=None, help="Binary threshold. Defaults to Otsu.")
    parser.add_argument("--threshold_bias", type=int, default=0, help="Bias applied after threshold estimation. Negative preserves more white gaps.")
    parser.add_argument("--invert", action="store_true", help="Treat light foreground on dark background as foreground.")
    parser.add_argument("--disable_alpha_foreground", action="store_true", help="Ignore PNG alpha when building the foreground field.")
    parser.add_argument("--preblur_radius", type=float, default=1.0, help="Blur radius applied before thresholding to stabilize anti-aliased edges.")
    parser.add_argument("--color_preblur_radius", type=float, default=0.0, help="Optional blur radius before palette quantization.")
    parser.add_argument("--morph_open_iterations", type=int, default=0, help="Optional binary opening passes before contour extraction.")
    parser.add_argument("--morph_close_iterations", type=int, default=0, help="Optional binary closing passes before contour extraction.")
    parser.add_argument("--max_colors", type=int, default=8, help="Maximum number of flat colors to keep during quantization.")
    parser.add_argument("--color_distance_threshold", type=float, default=28.0, help="Merge palette colors closer than this RGB distance.")
    parser.add_argument("--quantization_method", type=str, default="median_cut", help="Palette quantization method: median_cut, max_coverage, fast_octree.")
    parser.add_argument("--disable_merge_similar_colors", action="store_true", help="Keep quantized palette entries separate even if they are very close.")
    parser.add_argument("--background_mode", type=str, default="auto", help="Background handling: auto, manual, or none.")
    parser.add_argument("--background_color", type=str, default=None, help="Manual background color when background_mode=manual.")
    parser.add_argument("--min_region_area", type=int, default=32, help="Drop connected regions smaller than this.")
    parser.add_argument("--min_hole_area", type=int, default=16, help="Drop hole regions smaller than this.")
    parser.add_argument("--simplify_tolerance", type=float, default=1.25, help="Contour simplification tolerance.")
    parser.add_argument("--merge_distance", type=float, default=1.0, help="Merge contour points closer than this.")
    parser.add_argument("--corner_angle", type=float, default=115.0, help="Angles below this are protected during cleanup.")
    parser.add_argument("--smooth_iterations", type=int, default=2, help="Corner-aware smoothing passes on curved spans.")
    parser.add_argument("--smooth_strength", type=float, default=0.35, help="How strongly curved spans are smoothed.")
    parser.add_argument("--min_corner_angle_deg", type=float, default=124.0, help="Maximum angle still treated as a corner candidate during fitting.")
    parser.add_argument("--sharp_corner_angle_deg", type=float, default=92.0, help="Angles below this stay sharp when tip preservation is enabled.")
    parser.add_argument("--smooth_curve_angle_threshold", type=float, default=144.0, help="Angles above this are treated as smooth continuation, not corners.")
    parser.add_argument("--fillet_radius_ratio", type=float, default=0.11, help="Base fillet radius as a fraction of the shorter adjacent segment.")
    parser.add_argument("--max_fillet_radius", type=float, default=3.25, help="Upper bound for any automatically inserted fillet radius.")
    parser.add_argument("--preserve_tip_points", action="store_true", default=True, help="Preserve very acute tips as sharp corners.")
    parser.add_argument("--allow_tip_rounding", action="store_true", help="Allow even acute tips to be rounded when a stable fillet can be fit.")
    parser.add_argument("--disable_gap_preservation", action="store_true", help="Allow cleanup to collapse narrow negative-space gaps.")
    parser.add_argument("--gap_preservation_distance", type=float, default=10.0, help="Protect contour points across narrow negative-space gaps.")
    parser.add_argument("--gap_protection_span", type=int, default=2, help="How many neighboring points to protect around a narrow gap.")
    parser.add_argument("--disable_subpixel_contours", action="store_true", help="Fallback to pixel-edge contour extraction.")
    parser.add_argument("--primitive_fit_error_threshold", type=float, default=1.35, help="Maximum boundary error allowed for rectangle-like primitive fits.")
    parser.add_argument("--topology_area_tolerance", type=float, default=0.18, help="Maximum allowed relative area distortion before a fitted contour falls back.")
    parser.add_argument("--hole_area_tolerance", type=float, default=0.22, help="Maximum allowed relative area distortion for hole contours before fallback.")
    parser.add_argument("--topology_bbox_iou_threshold", type=float, default=0.55, help="Minimum bbox IoU required between fitted outer contours and source contours.")
    parser.add_argument("--hole_bbox_iou_threshold", type=float, default=0.42, help="Minimum bbox IoU required between fitted holes and source holes.")
    parser.add_argument("--line_fit_tolerance", type=float, default=1.0, help="Max line deviation before using a curve.")
    parser.add_argument("--arc_fit_tolerance", type=float, default=0.08, help="Maximum normalized error for promoting a span to a circular or elliptical arc.")
    parser.add_argument("--arc_confidence_threshold", type=float, default=0.78, help="Minimum confidence required before exporting a fitted arc segment.")
    parser.add_argument("--min_arc_sweep_degrees", type=float, default=18.0, help="Minimum sweep angle before a curved span is emitted as an arc.")
    parser.add_argument("--quadratic_fit_tolerance", type=float, default=1.35, help="Max fit error for quadratic Bezier spans.")
    parser.add_argument("--bezier_fit_tolerance", type=float, default=1.75, help="Max fit error for cubic Bezier spans.")
    parser.add_argument("--circle_fit_tolerance", type=float, default=0.08, help="Tolerance for promoting a contour to a circle.")
    parser.add_argument("--circle_circularity_min", type=float, default=0.78, help="Minimum circularity required before exporting a real circle.")
    parser.add_argument("--circle_confidence_threshold", type=float, default=0.86, help="Minimum confidence required before promoting a contour to a circle.")
    parser.add_argument("--ellipse_fit_tolerance", type=float, default=0.12, help="Tolerance for promoting a contour to an ellipse.")
    parser.add_argument("--ellipse_confidence_threshold", type=float, default=0.82, help="Minimum confidence required before promoting a contour to an ellipse.")
    parser.add_argument("--rectangle_confidence_threshold", type=float, default=0.88, help="Minimum confidence required before promoting a contour to a rectangle.")
    parser.add_argument("--rounded_rectangle_confidence_threshold", type=float, default=0.86, help="Minimum confidence required before promoting a contour to a rounded rectangle.")
    parser.add_argument("--fill", type=str, default="#000000", help="SVG fill color.")
    parser.add_argument("--stroke", type=str, default=None, help="Optional SVG stroke color.")
    parser.add_argument("--background", type=str, default=None, help="Optional SVG background fill.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = REPO_ROOT / input_path
    if not input_path.exists():
        raise FileNotFoundError(f"input image not found: {input_path}")

    output_path = Path(args.output) if args.output else _default_output_path(input_path)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path

    config = HybridVectorizerConfig(
        vectorization_mode=args.mode,
        target_size=args.target_size,
        threshold=args.threshold,
        threshold_bias=args.threshold_bias,
        invert=bool(args.invert),
        use_alpha_foreground=not args.disable_alpha_foreground,
        preblur_radius=args.preblur_radius,
        color_preblur_radius=args.color_preblur_radius,
        morph_open_iterations=args.morph_open_iterations,
        morph_close_iterations=args.morph_close_iterations,
        max_colors=args.max_colors,
        color_distance_threshold=args.color_distance_threshold,
        quantization_method=args.quantization_method,
        merge_similar_colors=not args.disable_merge_similar_colors,
        background_mode=args.background_mode,
        background_color=args.background_color,
        min_region_area=args.min_region_area,
        min_hole_area=args.min_hole_area,
        simplify_tolerance=args.simplify_tolerance,
        merge_distance=args.merge_distance,
        corner_angle_threshold_degrees=args.corner_angle,
        smooth_iterations=args.smooth_iterations,
        smooth_strength=args.smooth_strength,
        min_corner_angle_deg=args.min_corner_angle_deg,
        sharp_corner_angle_deg=args.sharp_corner_angle_deg,
        smooth_curve_angle_threshold=args.smooth_curve_angle_threshold,
        fillet_radius_ratio=args.fillet_radius_ratio,
        max_fillet_radius=args.max_fillet_radius,
        preserve_tip_points=bool(args.preserve_tip_points and not args.allow_tip_rounding),
        preserve_narrow_gaps=not args.disable_gap_preservation,
        gap_preservation_distance=args.gap_preservation_distance,
        gap_protection_span=args.gap_protection_span,
        subpixel_contours=not args.disable_subpixel_contours,
        primitive_fit_error_threshold=args.primitive_fit_error_threshold,
        topology_area_tolerance=args.topology_area_tolerance,
        hole_area_tolerance=args.hole_area_tolerance,
        topology_bbox_iou_threshold=args.topology_bbox_iou_threshold,
        hole_bbox_iou_threshold=args.hole_bbox_iou_threshold,
        line_fit_tolerance=args.line_fit_tolerance,
        arc_fit_tolerance=args.arc_fit_tolerance,
        arc_confidence_threshold=args.arc_confidence_threshold,
        min_arc_sweep_degrees=args.min_arc_sweep_degrees,
        quadratic_fit_tolerance=args.quadratic_fit_tolerance,
        bezier_fit_tolerance=args.bezier_fit_tolerance,
        circle_fit_tolerance=args.circle_fit_tolerance,
        circle_circularity_min=args.circle_circularity_min,
        circle_confidence_threshold=args.circle_confidence_threshold,
        ellipse_fit_tolerance=args.ellipse_fit_tolerance,
        ellipse_confidence_threshold=args.ellipse_confidence_threshold,
        rectangle_confidence_threshold=args.rectangle_confidence_threshold,
        rounded_rectangle_confidence_threshold=args.rounded_rectangle_confidence_threshold,
        fill_color=args.fill,
        stroke_color=args.stroke,
        background=args.background,
    )
    pipeline = HybridVectorizerPipeline(config)
    result = pipeline.vectorize(input_path, output_path=output_path)
    stats = document_stats(result.document)

    print(f"Input image: {input_path}")
    print(f"Output SVG: {output_path}")
    print(f"Threshold used: {result.preprocessed.threshold}")
    print(f"Shapes: {stats['num_shapes']}")
    print(f"Holes: {stats['num_holes']}")
    print(f"Loops: {stats['num_loops']}")
    print(f"Segments: {stats['num_segments']}")
    print(f"Layers: {stats['num_layers']}")
    print(f"Colors: {stats['num_colors']}")


if __name__ == "__main__":
    main()
