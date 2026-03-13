from __future__ import annotations

from yd_vector.hybrid_vectorizer.geometry import Shape, VectorDocument
from yd_vector.hybrid_vectorizer.svg_export import export_svg as export_svg_v1


def assemble_document(width: int, height: int, shapes: list[Shape], metadata: dict[str, str] | None = None) -> VectorDocument:
    return VectorDocument(
        width=width,
        height=height,
        shapes=shapes,
        metadata=metadata or {},
    )


def export_svg(document: VectorDocument, background: str | None = None) -> str:
    return export_svg_v1(document, background=background)
