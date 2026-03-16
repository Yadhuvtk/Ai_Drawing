from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from yd_vector.hybrid_vectorizer.color_segmentation import ColorSegmentationResult, segment_color_regions
from yd_vector.hybrid_vectorizer.cleanup import cleanup_region
from yd_vector.hybrid_vectorizer.config import HybridVectorizerConfig
from yd_vector.hybrid_vectorizer.contour_extraction import extract_regions
from yd_vector.hybrid_vectorizer.fitting import fit_region
from yd_vector.hybrid_vectorizer.geometry import (
    ClosedContour,
    ContourRegion,
    Loop,
    Point,
    PrimitiveCircle,
    PrimitiveEllipse,
    PrimitiveRectangle,
    PrimitiveRoundedRectangle,
    SegmentArcCircular,
    SegmentArcElliptical,
    SegmentBezierCubic,
    SegmentBezierQuadratic,
    SegmentLine,
    Shape,
    VectorDocument,
    VectorLayer,
    bounding_box_from_points,
    polygon_area,
)
from yd_vector.hybrid_vectorizer.preprocessing import PreprocessResult, preprocess_image
from yd_vector.hybrid_vectorizer.svg_export import export_svg


@dataclass
class HybridVectorizationResult:
    input_path: Path
    preprocessed: PreprocessResult
    segmentation: ColorSegmentationResult | None
    regions: list[ContourRegion]
    document: VectorDocument
    svg_text: str


class HybridVectorizerPipeline:
    def __init__(self, config: HybridVectorizerConfig | None = None) -> None:
        self.config = config or HybridVectorizerConfig()

    def vectorize(self, image_path: str | Path, output_path: str | Path | None = None) -> HybridVectorizationResult:
        input_path = Path(image_path)
        preprocessed = preprocess_image(input_path, self.config)
        segmentation: ColorSegmentationResult | None = None
        regions: list[ContourRegion] = []
        shapes: list[Shape] = []
        layers: list[VectorLayer] = []

        if self._should_use_color_vectorization():
            segmentation = segment_color_regions(preprocessed, self.config)
            regions, shapes, layers = self._vectorize_color_regions(segmentation)

        if not shapes:
            regions, shapes = self._vectorize_monochrome(preprocessed)
            layers = []

        if _needs_coordinate_rescale(preprocessed):
            regions = _scale_regions(
                regions,
                scale_x=preprocessed.coordinate_scale_x,
                scale_y=preprocessed.coordinate_scale_y,
            )
            if layers:
                layers = _scale_layers(
                    layers,
                    scale_x=preprocessed.coordinate_scale_x,
                    scale_y=preprocessed.coordinate_scale_y,
                )
                shapes = [shape for layer in layers for shape in layer.shapes]
            else:
                shapes = _scale_shapes(
                    shapes,
                    scale_x=preprocessed.coordinate_scale_x,
                    scale_y=preprocessed.coordinate_scale_y,
                )

        is_color_mode = segmentation is not None and layers
        metadata = {
            "pipeline": "hybrid_vectorizer_flat_color" if is_color_mode else "hybrid_vectorizer_monochrome",
            "threshold": str(preprocessed.threshold),
            "mode": "color" if is_color_mode else "monochrome",
            "max_colors": str(self.config.max_colors),
        }
        if segmentation is not None:
            metadata["palette"] = ",".join(entry.hex_color for entry in segmentation.palette if not entry.is_background)

        document = VectorDocument(
            width=preprocessed.original_width,
            height=preprocessed.original_height,
            shapes=shapes,
            layers=layers,
            metadata=metadata,
        )
        svg_text = export_svg(document, background=self.config.background)

        if output_path is not None:
            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(svg_text, encoding="utf-8", newline="\n")

        return HybridVectorizationResult(
            input_path=input_path,
            preprocessed=preprocessed,
            segmentation=segmentation,
            regions=regions,
            document=document,
            svg_text=svg_text,
        )

    def _should_use_color_vectorization(self) -> bool:
        mode = self.config.vectorization_mode.strip().lower()
        return mode in {"auto", "flat_color", "color"}

    def _vectorize_monochrome(self, preprocessed: PreprocessResult) -> tuple[list[ContourRegion], list[Shape]]:
        raw_regions = extract_regions(
            preprocessed.binary_mask,
            min_region_area=self.config.min_region_area,
            min_hole_area=self.config.min_hole_area,
            scalar_field=preprocessed.foreground,
            contour_level=preprocessed.foreground_threshold,
            subpixel=self.config.subpixel_contours,
        )
        regions = [cleanup_region(region, self.config) for region in raw_regions]
        shapes = [fit_region(region, self.config) for region in regions]
        return regions, shapes

    def _vectorize_color_regions(
        self,
        segmentation: ColorSegmentationResult,
    ) -> tuple[list[ContourRegion], list[Shape], list[VectorLayer]]:
        all_regions: list[ContourRegion] = []
        all_shapes: list[Shape] = []
        layers: list[VectorLayer] = []

        for layer_index, region_spec in enumerate(segmentation.color_regions):
            raw_regions = extract_regions(
                region_spec.mask,
                min_region_area=self.config.min_region_area,
                min_hole_area=self.config.min_hole_area,
                scalar_field=region_spec.scalar_field,
                contour_level=0.5,
                subpixel=self.config.subpixel_contours,
            )
            cleaned_regions = [cleanup_region(region, self.config) for region in raw_regions]
            cleaned_regions.sort(key=lambda item: item.outer.area, reverse=True)
            layer_shapes = []
            for shape_index, region in enumerate(cleaned_regions):
                layer_shapes.append(
                    fit_region(
                        region,
                        self.config,
                        fill_color=region_spec.fill_color,
                        shape_id=f"{region_spec.layer_id}_shape_{shape_index:02d}",
                        layer_id=region_spec.layer_id,
                        z_index=layer_index,
                    )
                )
            if not layer_shapes:
                continue

            layers.append(
                VectorLayer(
                    layer_id=region_spec.layer_id,
                    shapes=layer_shapes,
                    fill=region_spec.fill_color,
                    z_index=layer_index,
                )
            )
            all_regions.extend(cleaned_regions)
            all_shapes.extend(layer_shapes)

        return all_regions, all_shapes, layers


def _needs_coordinate_rescale(preprocessed: PreprocessResult) -> bool:
    return (
        abs(preprocessed.coordinate_scale_x - 1.0) > 1e-6
        or abs(preprocessed.coordinate_scale_y - 1.0) > 1e-6
    )


def _scale_regions(regions: list[ContourRegion], scale_x: float, scale_y: float) -> list[ContourRegion]:
    return [
        ContourRegion(
            region_id=region.region_id,
            outer=_scale_closed_contour(region.outer, scale_x, scale_y),
            holes=[_scale_closed_contour(hole, scale_x, scale_y) for hole in region.holes],
        )
        for region in regions
    ]


def _scale_shapes(shapes: list[Shape], scale_x: float, scale_y: float) -> list[Shape]:
    return [_scale_shape(shape, scale_x, scale_y) for shape in shapes]


def _scale_layers(layers: list[VectorLayer], scale_x: float, scale_y: float) -> list[VectorLayer]:
    return [
        VectorLayer(
            layer_id=layer.layer_id,
            shapes=[_scale_shape(shape, scale_x, scale_y) for shape in layer.shapes],
            fill=layer.fill,
            z_index=layer.z_index,
        )
        for layer in layers
    ]


def _scale_closed_contour(contour: ClosedContour, scale_x: float, scale_y: float) -> ClosedContour:
    points = [_scale_point(point, scale_x, scale_y) for point in contour.points]
    return ClosedContour(
        contour_id=contour.contour_id,
        points=points,
        area=abs(polygon_area(points)),
        bbox=bounding_box_from_points(points),
        is_hole=contour.is_hole,
        parent_id=contour.parent_id,
        children_ids=contour.children_ids[:],
    )


def _scale_shape(shape: Shape, scale_x: float, scale_y: float) -> Shape:
    return Shape(
        shape_id=shape.shape_id,
        outer_loop=_scale_loop(shape.outer_loop, scale_x, scale_y),
        negative_loops=[_scale_loop(loop, scale_x, scale_y) for loop in shape.negative_loops],
        fill=shape.fill,
        stroke=shape.stroke,
        layer_id=shape.layer_id,
        z_index=shape.z_index,
    )


def _scale_loop(loop: Loop, scale_x: float, scale_y: float) -> Loop:
    return Loop(
        loop_id=loop.loop_id,
        segments=[_scale_segment(segment, scale_x, scale_y) for segment in loop.segments],
        polarity=loop.polarity,
        closed=loop.closed,
        source_contour_id=loop.source_contour_id,
        primitive=_scale_primitive(loop.primitive, scale_x, scale_y),
        confidence=loop.confidence,
    )


def _scale_segment(
    segment: SegmentLine | SegmentArcCircular | SegmentArcElliptical | SegmentBezierQuadratic | SegmentBezierCubic,
    scale_x: float,
    scale_y: float,
) -> SegmentLine | SegmentArcCircular | SegmentArcElliptical | SegmentBezierQuadratic | SegmentBezierCubic:
    if isinstance(segment, SegmentLine):
        return SegmentLine(
            start=_scale_point(segment.start, scale_x, scale_y),
            end=_scale_point(segment.end, scale_x, scale_y),
        )
    if isinstance(segment, SegmentArcCircular):
        uniform_scale = _uniform_scale(scale_x, scale_y)
        return SegmentArcCircular(
            start=_scale_point(segment.start, scale_x, scale_y),
            end=_scale_point(segment.end, scale_x, scale_y),
            radius=segment.radius * uniform_scale,
            large_arc=segment.large_arc,
            sweep=segment.sweep,
        )
    if isinstance(segment, SegmentArcElliptical):
        return SegmentArcElliptical(
            start=_scale_point(segment.start, scale_x, scale_y),
            end=_scale_point(segment.end, scale_x, scale_y),
            radius_x=segment.radius_x * scale_x,
            radius_y=segment.radius_y * scale_y,
            rotation_degrees=segment.rotation_degrees,
            large_arc=segment.large_arc,
            sweep=segment.sweep,
        )
    if isinstance(segment, SegmentBezierQuadratic):
        return SegmentBezierQuadratic(
            start=_scale_point(segment.start, scale_x, scale_y),
            control=_scale_point(segment.control, scale_x, scale_y),
            end=_scale_point(segment.end, scale_x, scale_y),
        )
    return SegmentBezierCubic(
        start=_scale_point(segment.start, scale_x, scale_y),
        control1=_scale_point(segment.control1, scale_x, scale_y),
        control2=_scale_point(segment.control2, scale_x, scale_y),
        end=_scale_point(segment.end, scale_x, scale_y),
    )


def _scale_primitive(
    primitive: PrimitiveCircle | PrimitiveEllipse | PrimitiveRectangle | PrimitiveRoundedRectangle | None,
    scale_x: float,
    scale_y: float,
) -> PrimitiveCircle | PrimitiveEllipse | PrimitiveRectangle | PrimitiveRoundedRectangle | None:
    if primitive is None:
        return None
    if isinstance(primitive, PrimitiveCircle):
        center = _scale_point(primitive.center, scale_x, scale_y)
        if abs(scale_x - scale_y) <= 1e-6:
            return PrimitiveCircle(center=center, radius=primitive.radius * scale_x)
        return PrimitiveEllipse(
            center=center,
            radius_x=primitive.radius * scale_x,
            radius_y=primitive.radius * scale_y,
            rotation_degrees=0.0,
        )
    if isinstance(primitive, PrimitiveEllipse):
        return PrimitiveEllipse(
            center=_scale_point(primitive.center, scale_x, scale_y),
            radius_x=primitive.radius_x * scale_x,
            radius_y=primitive.radius_y * scale_y,
            rotation_degrees=primitive.rotation_degrees,
        )
    if isinstance(primitive, PrimitiveRectangle):
        return PrimitiveRectangle(
            center=_scale_point(primitive.center, scale_x, scale_y),
            width=primitive.width * scale_x,
            height=primitive.height * scale_y,
            rotation_degrees=primitive.rotation_degrees,
        )
    return PrimitiveRoundedRectangle(
        center=_scale_point(primitive.center, scale_x, scale_y),
        width=primitive.width * scale_x,
        height=primitive.height * scale_y,
        corner_radius=primitive.corner_radius * _uniform_scale(scale_x, scale_y),
        rotation_degrees=primitive.rotation_degrees,
    )


def _scale_point(point: Point, scale_x: float, scale_y: float) -> Point:
    return Point(point.x * scale_x, point.y * scale_y)


def _uniform_scale(scale_x: float, scale_y: float) -> float:
    return (scale_x + scale_y) * 0.5


# TODO: Allow an optional AI-assisted structure extractor to replace or augment threshold-based region extraction.
