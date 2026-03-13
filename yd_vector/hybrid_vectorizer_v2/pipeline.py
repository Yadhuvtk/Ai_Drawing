from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from yd_vector.hybrid_vectorizer.geometry import VectorDocument
from yd_vector.hybrid_vectorizer_v2.assembler import assemble_region_shape
from yd_vector.hybrid_vectorizer_v2.config import HybridVectorizerV2Config
from yd_vector.hybrid_vectorizer_v2.decompose import decompose_region
from yd_vector.hybrid_vectorizer_v2.geometry import RegionDecomposition
from yd_vector.hybrid_vectorizer_v2.structure import ExtractedStructure, extract_monochrome_structure
from yd_vector.hybrid_vectorizer_v2.svg_assembler import assemble_document, export_svg


@dataclass
class HybridVectorizationV2Result:
    input_path: Path
    structure: ExtractedStructure
    decompositions: list[RegionDecomposition]
    document: VectorDocument
    svg_text: str


class PrimitiveFirstVectorizerV2:
    def __init__(self, config: HybridVectorizerV2Config | None = None) -> None:
        self.config = config or HybridVectorizerV2Config()

    def vectorize(self, image_path: str | Path, output_path: str | Path | None = None) -> HybridVectorizationV2Result:
        structure = extract_monochrome_structure(image_path, self.config)
        decompositions = [decompose_region(region, self.config) for region in structure.regions]
        shapes = [assemble_region_shape(item, self.config) for item in decompositions]
        metadata = {
            "pipeline": "hybrid_vectorizer_v2",
            "mode": "monochrome",
            "architecture": "primitive_first_topology_safe",
            "threshold": str(structure.preprocessed.threshold),
        }
        document = assemble_document(
            width=structure.preprocessed.width,
            height=structure.preprocessed.height,
            shapes=shapes,
            metadata=metadata,
        )
        svg_text = export_svg(document, background=self.config.background)

        if output_path is not None:
            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(svg_text, encoding="utf-8", newline="\n")

        return HybridVectorizationV2Result(
            input_path=Path(image_path),
            structure=structure,
            decompositions=decompositions,
            document=document,
            svg_text=svg_text,
        )
