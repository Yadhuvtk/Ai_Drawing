from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from yd_vector.hybrid_vectorizer.color_segmentation import ColorSegmentationResult, segment_color_regions
from yd_vector.hybrid_vectorizer.cleanup import cleanup_region
from yd_vector.hybrid_vectorizer.config import HybridVectorizerConfig
from yd_vector.hybrid_vectorizer.contour_extraction import extract_regions
from yd_vector.hybrid_vectorizer.fitting import fit_region
from yd_vector.hybrid_vectorizer.geometry import ContourRegion, Shape, VectorDocument, VectorLayer
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
            width=preprocessed.width,
            height=preprocessed.height,
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


# TODO: Allow an optional AI-assisted structure extractor to replace or augment threshold-based region extraction.
