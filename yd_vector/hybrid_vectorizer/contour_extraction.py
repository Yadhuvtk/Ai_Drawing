from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from yd_vector.hybrid_vectorizer.geometry import (
    ClosedContour,
    ContourRegion,
    Point,
    bounding_box_from_points,
    polygon_area,
)


GridPoint = tuple[int, int]
FloatPoint = tuple[float, float]
Pixel = tuple[int, int]


@dataclass(frozen=True)
class ComponentPatch:
    mask: np.ndarray
    origin_x: int
    origin_y: int
    scalar_field: np.ndarray | None = None


def extract_regions(
    mask: np.ndarray,
    min_region_area: int = 32,
    min_hole_area: int = 16,
    scalar_field: np.ndarray | None = None,
    contour_level: float | None = None,
    subpixel: bool = True,
) -> list[ContourRegion]:
    mask = np.asarray(mask, dtype=bool)
    scalar_field = None if scalar_field is None else np.asarray(scalar_field, dtype=np.float32)
    regions: list[ContourRegion] = []

    region_index = 0
    for component_pixels in _connected_components(mask):
        if len(component_pixels) < min_region_area:
            continue

        region_id = f"region_{region_index:03d}"
        component = _component_patch(component_pixels, mask.shape, scalar_field=scalar_field, padding=1)
        outer = _mask_to_contour(
            component.mask,
            origin_x=component.origin_x,
            origin_y=component.origin_y,
            contour_id=f"{region_id}_outer",
            is_hole=False,
            scalar_field=component.scalar_field,
            level=contour_level if contour_level is not None else 0.5,
            subpixel=subpixel,
        )

        holes: list[ClosedContour] = []
        for hole_index, hole_pixels in enumerate(_hole_components(component.mask)):
            if len(hole_pixels) < min_hole_area:
                continue

            hole_patch = _component_patch(hole_pixels, component.mask.shape, padding=1)
            hole = _mask_to_contour(
                hole_patch.mask,
                origin_x=component.origin_x + hole_patch.origin_x,
                origin_y=component.origin_y + hole_patch.origin_y,
                contour_id=f"{region_id}_hole_{hole_index:02d}",
                is_hole=True,
                parent_id=outer.contour_id,
                scalar_field=None,
                level=0.5,
                subpixel=subpixel,
            )
            holes.append(hole)

        outer.children_ids = [hole.contour_id for hole in holes]
        holes.sort(key=lambda hole: hole.area, reverse=True)
        regions.append(ContourRegion(region_id=region_id, outer=outer, holes=holes))
        region_index += 1

    regions.sort(key=lambda region: region.outer.area, reverse=True)
    return regions


def _connected_components(mask: np.ndarray) -> list[list[Pixel]]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[list[Pixel]] = []

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue

            queue = deque([(y, x)])
            visited[y, x] = True
            pixels: list[Pixel] = []
            while queue:
                cy, cx = queue.popleft()
                pixels.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            components.append(pixels)
    return components


def _component_patch(
    component_pixels: list[Pixel],
    base_shape: tuple[int, int],
    scalar_field: np.ndarray | None = None,
    padding: int = 1,
) -> ComponentPatch:
    base_height, base_width = base_shape
    raw_min_y = min(y for y, _ in component_pixels) - padding
    raw_max_y = max(y for y, _ in component_pixels) + padding
    raw_min_x = min(x for _, x in component_pixels) - padding
    raw_max_x = max(x for _, x in component_pixels) + padding

    patch_height = raw_max_y - raw_min_y + 1
    patch_width = raw_max_x - raw_min_x + 1
    mask = np.zeros((patch_height, patch_width), dtype=bool)
    for y, x in component_pixels:
        mask[y - raw_min_y, x - raw_min_x] = True

    local_field = None
    if scalar_field is not None:
        local_field = np.zeros((patch_height, patch_width), dtype=np.float32)
        src_y0 = max(raw_min_y, 0)
        src_y1 = min(raw_max_y + 1, base_height)
        src_x0 = max(raw_min_x, 0)
        src_x1 = min(raw_max_x + 1, base_width)
        dst_y0 = src_y0 - raw_min_y
        dst_x0 = src_x0 - raw_min_x
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        local_field[dst_y0:dst_y1, dst_x0:dst_x1] = scalar_field[src_y0:src_y1, src_x0:src_x1]

    return ComponentPatch(mask=mask, origin_x=raw_min_x, origin_y=raw_min_y, scalar_field=local_field)


def _hole_components(component_mask: np.ndarray) -> list[list[Pixel]]:
    inverse = ~component_mask
    height, width = inverse.shape
    exterior = np.zeros_like(inverse, dtype=bool)
    queue: deque[Pixel] = deque()

    for x in range(width):
        if inverse[0, x]:
            exterior[0, x] = True
            queue.append((0, x))
        if inverse[height - 1, x] and not exterior[height - 1, x]:
            exterior[height - 1, x] = True
            queue.append((height - 1, x))
    for y in range(height):
        if inverse[y, 0] and not exterior[y, 0]:
            exterior[y, 0] = True
            queue.append((y, 0))
        if inverse[y, width - 1] and not exterior[y, width - 1]:
            exterior[y, width - 1] = True
            queue.append((y, width - 1))

    while queue:
        cy, cx = queue.popleft()
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if 0 <= ny < height and 0 <= nx < width and inverse[ny, nx] and not exterior[ny, nx]:
                exterior[ny, nx] = True
                queue.append((ny, nx))

    hole_mask = inverse & ~exterior
    return _connected_components(hole_mask)


def _mask_to_contour(
    mask: np.ndarray,
    origin_x: int,
    origin_y: int,
    contour_id: str,
    is_hole: bool,
    parent_id: str | None = None,
    scalar_field: np.ndarray | None = None,
    level: float = 0.5,
    subpixel: bool = True,
) -> ClosedContour:
    points: list[Point]
    if subpixel:
        field = scalar_field if scalar_field is not None else mask.astype(np.float32)
        loops = _marching_square_loops(field, level if scalar_field is not None else 0.5)
        if loops:
            loop = _select_primary_loop(loops)
            points = [Point(origin_x + x + 0.5, origin_y + y + 0.5) for x, y in loop]
        else:
            points = _fallback_boundary_points(mask, origin_x, origin_y)
    else:
        points = _fallback_boundary_points(mask, origin_x, origin_y)

    points = _dedupe_consecutive_points(points)
    if len(points) < 3:
        raise ValueError(f"No boundary loops extracted for contour {contour_id}")

    signed_area = polygon_area(points)
    if is_hole and signed_area > 0.0:
        points = list(reversed(points))
    elif not is_hole and signed_area < 0.0:
        points = list(reversed(points))

    area = abs(polygon_area(points))
    bbox = bounding_box_from_points(points)
    return ClosedContour(
        contour_id=contour_id,
        points=points,
        area=area,
        bbox=bbox,
        is_hole=is_hole,
        parent_id=parent_id,
    )


def _marching_square_loops(field: np.ndarray, level: float) -> list[list[FloatPoint]]:
    if field.shape[0] < 2 or field.shape[1] < 2:
        return []
    segments = _marching_square_segments(field, level)
    return _assemble_segment_loops(segments)


def _marching_square_segments(field: np.ndarray, level: float) -> list[tuple[FloatPoint, FloatPoint]]:
    height, width = field.shape
    segments: list[tuple[FloatPoint, FloatPoint]] = []
    for y in range(height - 1):
        for x in range(width - 1):
            v0 = float(field[y, x])
            v1 = float(field[y, x + 1])
            v2 = float(field[y + 1, x + 1])
            v3 = float(field[y + 1, x])

            cell_mask = 0
            if v0 >= level:
                cell_mask |= 1
            if v1 >= level:
                cell_mask |= 2
            if v2 >= level:
                cell_mask |= 4
            if v3 >= level:
                cell_mask |= 8
            if cell_mask in (0, 15):
                continue

            intersections = {
                "top": _interpolate((x, y), (x + 1, y), v0, v1, level),
                "right": _interpolate((x + 1, y), (x + 1, y + 1), v1, v2, level),
                "bottom": _interpolate((x, y + 1), (x + 1, y + 1), v3, v2, level),
                "left": _interpolate((x, y), (x, y + 1), v0, v3, level),
            }
            center_inside = ((v0 + v1 + v2 + v3) * 0.25) >= level
            for edge_a, edge_b in _case_segments(cell_mask, center_inside):
                segments.append((intersections[edge_a], intersections[edge_b]))
    return segments


def _case_segments(cell_mask: int, center_inside: bool) -> list[tuple[str, str]]:
    lookup: dict[int, list[tuple[str, str]]] = {
        1: [("left", "top")],
        2: [("top", "right")],
        3: [("left", "right")],
        4: [("right", "bottom")],
        5: [("left", "top"), ("right", "bottom")],
        6: [("top", "bottom")],
        7: [("left", "bottom")],
        8: [("bottom", "left")],
        9: [("top", "bottom")],
        10: [("top", "right"), ("bottom", "left")],
        11: [("right", "bottom")],
        12: [("left", "right")],
        13: [("top", "right")],
        14: [("left", "top")],
    }
    if cell_mask == 5:
        return [("top", "right"), ("bottom", "left")] if center_inside else lookup[cell_mask]
    if cell_mask == 10:
        return lookup[cell_mask] if center_inside else [("left", "top"), ("right", "bottom")]
    return lookup.get(cell_mask, [])


def _interpolate(
    start: tuple[int, int],
    end: tuple[int, int],
    start_value: float,
    end_value: float,
    level: float,
) -> FloatPoint:
    if abs(end_value - start_value) <= 1e-8:
        t = 0.5
    else:
        t = (level - start_value) / (end_value - start_value)
        t = max(0.0, min(1.0, t))
    return (
        float(start[0] + (end[0] - start[0]) * t),
        float(start[1] + (end[1] - start[1]) * t),
    )


def _assemble_segment_loops(segments: list[tuple[FloatPoint, FloatPoint]]) -> list[list[FloatPoint]]:
    if not segments:
        return []

    adjacency: dict[FloatPoint, list[FloatPoint]] = defaultdict(list)
    points_by_key: dict[FloatPoint, FloatPoint] = {}
    for start, end in segments:
        start_key = _round_point(start)
        end_key = _round_point(end)
        if start_key == end_key:
            continue
        points_by_key.setdefault(start_key, start)
        points_by_key.setdefault(end_key, end)
        adjacency[start_key].append(end_key)
        adjacency[end_key].append(start_key)

    visited_edges: set[tuple[FloatPoint, FloatPoint]] = set()
    loops: list[list[FloatPoint]] = []
    for start_key, neighbors in adjacency.items():
        for next_key in neighbors:
            edge_key = _edge_key(start_key, next_key)
            if edge_key in visited_edges:
                continue

            loop = [points_by_key[start_key]]
            prev_key = start_key
            current_key = next_key
            visited_edges.add(edge_key)

            while True:
                if current_key == start_key:
                    break
                loop.append(points_by_key[current_key])

                candidates = adjacency[current_key]
                next_candidate = None
                for candidate in candidates:
                    if candidate == prev_key and len(candidates) > 1:
                        continue
                    candidate_edge = _edge_key(current_key, candidate)
                    if candidate_edge in visited_edges and candidate != start_key:
                        continue
                    next_candidate = candidate
                    break

                if next_candidate is None:
                    loop = []
                    break

                next_edge = _edge_key(current_key, next_candidate)
                if next_edge in visited_edges and next_candidate != start_key:
                    loop = []
                    break

                visited_edges.add(next_edge)
                prev_key, current_key = current_key, next_candidate
                if len(loop) > len(adjacency) + 4:
                    loop = []
                    break

            if len(loop) >= 3:
                loops.append(_dedupe_float_loop(loop))
    return [loop for loop in loops if len(loop) >= 3]


def _round_point(point: FloatPoint, decimals: int = 6) -> FloatPoint:
    return (round(point[0], decimals), round(point[1], decimals))


def _edge_key(left: FloatPoint, right: FloatPoint) -> tuple[FloatPoint, FloatPoint]:
    return (left, right) if left <= right else (right, left)


def _dedupe_float_loop(points: list[FloatPoint]) -> list[FloatPoint]:
    if not points:
        return points
    deduped = [points[0]]
    for point in points[1:]:
        if abs(point[0] - deduped[-1][0]) > 1e-6 or abs(point[1] - deduped[-1][1]) > 1e-6:
            deduped.append(point)
    if len(deduped) > 2 and abs(deduped[0][0] - deduped[-1][0]) <= 1e-6 and abs(deduped[0][1] - deduped[-1][1]) <= 1e-6:
        deduped.pop()
    return deduped


def _dedupe_consecutive_points(points: list[Point]) -> list[Point]:
    if not points:
        return points
    deduped = [points[0]]
    for point in points[1:]:
        if abs(point.x - deduped[-1].x) > 1e-6 or abs(point.y - deduped[-1].y) > 1e-6:
            deduped.append(point)
    if len(deduped) > 2 and abs(deduped[0].x - deduped[-1].x) <= 1e-6 and abs(deduped[0].y - deduped[-1].y) <= 1e-6:
        deduped.pop()
    return deduped


def _select_primary_loop(loops: list[list[FloatPoint]]) -> list[FloatPoint]:
    def score(loop: list[FloatPoint]) -> tuple[float, int]:
        points = [Point(x, y) for x, y in loop]
        return (abs(polygon_area(points)), len(loop))

    return max(loops, key=score)


def _fallback_boundary_points(mask: np.ndarray, origin_x: int, origin_y: int) -> list[Point]:
    edges = _boundary_edges(mask)
    loops = _edge_loops(edges)
    if not loops:
        return []
    return [Point(origin_x + x, origin_y + y) for x, y in _select_grid_loop(loops)]


def _select_grid_loop(loops: list[list[GridPoint]]) -> list[GridPoint]:
    def score(loop: list[GridPoint]) -> tuple[float, int]:
        points = [Point(x, y) for x, y in loop]
        return (abs(polygon_area(points)), len(loop))

    return max(loops, key=score)


def _boundary_edges(mask: np.ndarray) -> list[tuple[GridPoint, GridPoint]]:
    edges: list[tuple[GridPoint, GridPoint]] = []
    height, width = mask.shape

    for y in range(height):
        for x in range(width):
            if not mask[y, x]:
                continue
            if y == 0 or not mask[y - 1, x]:
                edges.append(((x, y), (x + 1, y)))
            if x == width - 1 or not mask[y, x + 1]:
                edges.append(((x + 1, y), (x + 1, y + 1)))
            if y == height - 1 or not mask[y + 1, x]:
                edges.append(((x + 1, y + 1), (x, y + 1)))
            if x == 0 or not mask[y, x - 1]:
                edges.append(((x, y + 1), (x, y)))
    return edges


def _edge_loops(edges: list[tuple[GridPoint, GridPoint]]) -> list[list[GridPoint]]:
    outgoing: dict[GridPoint, list[GridPoint]] = defaultdict(list)
    for start, end in edges:
        outgoing[start].append(end)
    for values in outgoing.values():
        values.sort()

    loops: list[list[GridPoint]] = []
    while outgoing:
        start = min(outgoing.keys(), key=lambda point: (point[1], point[0]))
        current = start
        loop = [start]

        while True:
            next_points = outgoing[current]
            nxt = next_points.pop(0)
            if not next_points:
                del outgoing[current]
            current = nxt
            if current == start:
                break
            loop.append(current)
        loops.append(loop)
    return loops


# TODO: Replace hard thresholding with a true signed-distance or edge-energy field for harder raster inputs.
