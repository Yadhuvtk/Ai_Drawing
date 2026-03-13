from __future__ import annotations

import numpy as np

from yd_vector.hybrid_vectorizer.geometry import VectorDocument


def document_stats(document: VectorDocument) -> dict[str, int]:
    num_shapes = len(document.shapes)
    num_holes = sum(len(shape.negative_loops) for shape in document.shapes)
    num_loops = sum(1 + len(shape.negative_loops) for shape in document.shapes)
    num_segments = sum(
        len(shape.outer_loop.segments) + sum(len(loop.segments) for loop in shape.negative_loops)
        for shape in document.shapes
    )
    num_layers = len(document.layers)
    num_colors = len({shape.fill for shape in document.shapes})
    return {
        "num_shapes": num_shapes,
        "num_holes": num_holes,
        "num_loops": num_loops,
        "num_segments": num_segments,
        "num_primitives": num_segments,
        "num_layers": num_layers,
        "num_colors": num_colors,
    }


def mask_iou(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    pred = np.asarray(pred_mask, dtype=bool)
    target = np.asarray(target_mask, dtype=bool)
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return float(intersection / union) if union > 0 else 1.0


# TODO: Add rasterized SVG vs target-image metrics once an optional render-back evaluation loop is in place.
