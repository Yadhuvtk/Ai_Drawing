from __future__ import annotations

from dataclasses import replace

from yd_vector.hybrid_vectorizer.geometry import Loop, Shape


def assemble_shape(
    shape_id: str,
    outer_loop: Loop,
    negative_loops: list[Loop],
    fill: str,
    stroke: str | None,
    layer_id: str | None = None,
    z_index: int = 0,
) -> Shape:
    normalized_outer = replace(outer_loop, polarity="positive")
    normalized_negative = [replace(loop, polarity="negative") for loop in negative_loops]
    return Shape(
        shape_id=shape_id,
        outer_loop=normalized_outer,
        negative_loops=normalized_negative,
        fill=fill,
        stroke=stroke,
        layer_id=layer_id,
        z_index=z_index,
    )
