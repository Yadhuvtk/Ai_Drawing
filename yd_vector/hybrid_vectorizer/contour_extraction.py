from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from yd_vector.hybrid_vectorizer.geometry import (
    ClosedContour,
    ContourRegion,
    Point,
    bounding_box_from_points,
    polygon_area,
    polygon_perimeter,
)


@dataclass
class RegionLoop(ClosedContour):
    perimeter: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def loop_id(self) -> str:
        return self.contour_id


@dataclass
class RegionShape(ContourRegion):
    fill_polarity: str = "positive"
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def outer_loop(self) -> RegionLoop:
        return self.outer  # type: ignore[return-value]

    @property
    def hole_loops(self) -> list[RegionLoop]:
        return self.holes  # type: ignore[return-value]


def extract_regions(
    mask: np.ndarray,
    min_region_area: int = 32,
    min_hole_area: int = 16,
    scalar_field: np.ndarray | None = None,
    contour_level: float | None = None,
    subpixel: bool = True,
    coordinate_scale_x: float = 1.0,
    coordinate_scale_y: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> list[RegionShape]:
    del scalar_field, contour_level

    binary_mask = np.asarray(mask, dtype=np.uint8)
    if binary_mask.ndim != 2 or binary_mask.size == 0:
        return []

    binary_mask = np.where(binary_mask > 0, 255, 0).astype(np.uint8)
    contours, hierarchy = _find_contours_with_hierarchy(binary_mask)
    if hierarchy is None or not contours:
        return []

    hierarchy = hierarchy[0]
    depths = [_hierarchy_depth(index, hierarchy) for index in range(len(contours))]
    regions: list[RegionShape] = []
    region_index = 0

    for contour_index, contour in enumerate(contours):
        if depths[contour_index] % 2 != 0:
            continue

        outer = _contour_to_closed_contour(
            contour=contour,
            contour_id=f"region_{region_index:03d}_outer",
            is_hole=False,
            parent_id=None,
            subpixel=subpixel,
            coordinate_scale_x=coordinate_scale_x,
            coordinate_scale_y=coordinate_scale_y,
            offset_x=offset_x,
            offset_y=offset_y,
        )
        if outer is None or outer.area < min_region_area:
            continue

        holes: list[RegionLoop] = []
        child_index = int(hierarchy[contour_index][2])
        hole_index = 0
        while child_index != -1:
            if depths[child_index] == depths[contour_index] + 1:
                hole = _contour_to_closed_contour(
                    contour=contours[child_index],
                    contour_id=f"region_{region_index:03d}_hole_{hole_index:02d}",
                    is_hole=True,
                    parent_id=outer.contour_id,
                    subpixel=subpixel,
                    coordinate_scale_x=coordinate_scale_x,
                    coordinate_scale_y=coordinate_scale_y,
                    offset_x=offset_x,
                    offset_y=offset_y,
                )
                if hole is not None and hole.area >= min_hole_area:
                    holes.append(hole)
                    hole_index += 1
            child_index = int(hierarchy[child_index][0])

        outer.children_ids = [hole.contour_id for hole in holes]
        holes.sort(key=lambda item: item.area, reverse=True)
        regions.append(
            RegionShape(
                region_id=f"region_{region_index:03d}",
                outer=outer,
                holes=holes,
                fill_polarity="positive",
                metadata={
                    "source": "binary_mask",
                    "subpixel": bool(subpixel),
                    "coordinate_space": "image",
                    "coordinate_scale_x": float(coordinate_scale_x),
                    "coordinate_scale_y": float(coordinate_scale_y),
                    "offset_x": float(offset_x),
                    "offset_y": float(offset_y),
                },
            )
        )
        region_index += 1

    regions.sort(key=lambda item: item.outer.area, reverse=True)
    return regions


def extract_region_shapes(
    mask: np.ndarray,
    min_region_area: int = 32,
    min_hole_area: int = 16,
    scalar_field: np.ndarray | None = None,
    contour_level: float | None = None,
    subpixel: bool = True,
    coordinate_scale_x: float = 1.0,
    coordinate_scale_y: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> list[RegionShape]:
    return extract_regions(
        mask,
        min_region_area=min_region_area,
        min_hole_area=min_hole_area,
        scalar_field=scalar_field,
        contour_level=contour_level,
        subpixel=subpixel,
        coordinate_scale_x=coordinate_scale_x,
        coordinate_scale_y=coordinate_scale_y,
        offset_x=offset_x,
        offset_y=offset_y,
    )


def _find_contours_with_hierarchy(binary_mask: np.ndarray) -> tuple[list[np.ndarray], np.ndarray | None]:
    result = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(result) == 3:
        _, contours, hierarchy = result
    else:
        contours, hierarchy = result
    return list(contours), hierarchy


def _hierarchy_depth(index: int, hierarchy: np.ndarray) -> int:
    depth = 0
    parent_index = int(hierarchy[index][3])
    while parent_index != -1:
        depth += 1
        parent_index = int(hierarchy[parent_index][3])
    return depth


def _contour_to_closed_contour(
    contour: np.ndarray,
    contour_id: str,
    is_hole: bool,
    parent_id: str | None,
    subpixel: bool,
    coordinate_scale_x: float,
    coordinate_scale_y: float,
    offset_x: float,
    offset_y: float,
) -> RegionLoop | None:
    coords = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    if len(coords) < 3:
        return None

    offset = 0.5 if subpixel else 0.0
    points = _dedupe_consecutive_points(
        [
            Point(
                (float(x) + offset) * float(coordinate_scale_x) + float(offset_x),
                (float(y) + offset) * float(coordinate_scale_y) + float(offset_y),
            )
            for x, y in coords
        ]
    )
    if len(points) < 3:
        return None

    signed_area = polygon_area(points)
    if abs(signed_area) <= 1e-6:
        return None
    if is_hole and signed_area > 0.0:
        points = list(reversed(points))
    elif not is_hole and signed_area < 0.0:
        points = list(reversed(points))

    area = abs(polygon_area(points))
    return RegionLoop(
        contour_id=contour_id,
        points=points,
        area=area,
        bbox=bounding_box_from_points(points),
        perimeter=polygon_perimeter(points),
        is_hole=is_hole,
        parent_id=parent_id,
        metadata={
            "offset_x": float(offset_x),
            "offset_y": float(offset_y),
            "subpixel_offset": 0.5 if subpixel else 0.0,
            "coordinate_scale_x": float(coordinate_scale_x),
            "coordinate_scale_y": float(coordinate_scale_y),
            "coordinate_space": "image",
        },
    )


def _dedupe_consecutive_points(points: list[Point]) -> list[Point]:
    if not points:
        return []

    deduped = [points[0]]
    for point in points[1:]:
        if abs(point.x - deduped[-1].x) > 1e-6 or abs(point.y - deduped[-1].y) > 1e-6:
            deduped.append(point)
    if len(deduped) > 2 and abs(deduped[0].x - deduped[-1].x) <= 1e-6 and abs(deduped[0].y - deduped[-1].y) <= 1e-6:
        deduped.pop()
    return deduped


# TODO: Refine hierarchy-derived contours with scalar-field subpixel snapping for harder raster inputs.
