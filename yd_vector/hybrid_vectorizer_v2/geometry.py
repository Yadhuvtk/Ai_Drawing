from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from yd_vector.hybrid_vectorizer.geometry import (
    ClosedContour,
    ContourRegion,
    Loop,
    Point,
    PrimitiveCircle,
    PrimitiveEllipse,
    PrimitiveRectangle,
    PrimitiveRoundedRectangle,
    SegmentArcElliptical,
    Shape,
    VectorDocument,
)


PrimitiveKind = Literal["circle", "ellipse", "rectangle", "rounded_rectangle"]
ContourRole = Literal["outer", "hole"]
LoopPolarity = Literal["positive", "negative"]


@dataclass(frozen=True)
class StraightRunCandidate:
    start_anchor: int
    end_anchor: int
    mean_error: float
    max_error: float
    length: float


@dataclass(frozen=True)
class PrimitiveCandidate:
    kind: PrimitiveKind
    confidence: float
    score: float
    loop: Loop
    notes: str = ""


@dataclass(frozen=True)
class EllipseSubshapeCandidate:
    start_source_index: int
    end_source_index: int
    start_anchor_index: int
    end_anchor_index: int
    segment: SegmentArcElliptical
    confidence: float
    symmetry_axis_x: float
    notes: str = ""


@dataclass
class ContourPartPlan:
    contour: ClosedContour
    role: ContourRole
    polarity: LoopPolarity
    anchor_points: list[Point]
    anchor_indices: list[int]
    straight_runs: list[StraightRunCandidate] = field(default_factory=list)
    primitive_candidates: list[PrimitiveCandidate] = field(default_factory=list)
    ellipse_subshape: EllipseSubshapeCandidate | None = None


@dataclass
class RegionDecomposition:
    region: ContourRegion
    outer: ContourPartPlan
    holes: list[ContourPartPlan] = field(default_factory=list)


@dataclass
class HybridVectorizationV2Document:
    width: int
    height: int
    shapes: list[Shape]
    metadata: dict[str, str] = field(default_factory=dict)

    def to_vector_document(self) -> VectorDocument:
        return VectorDocument(
            width=self.width,
            height=self.height,
            shapes=self.shapes,
            metadata=self.metadata,
        )


PrimitiveShape = PrimitiveCircle | PrimitiveEllipse | PrimitiveRectangle | PrimitiveRoundedRectangle
