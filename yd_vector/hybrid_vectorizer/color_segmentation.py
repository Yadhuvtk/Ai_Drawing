from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from PIL import Image

from yd_vector.hybrid_vectorizer.config import HybridVectorizerConfig
from yd_vector.hybrid_vectorizer.preprocessing import PreprocessResult


MAX_COLOR_DISTANCE = float(sqrt(3.0 * 255.0 * 255.0))


@dataclass(frozen=True)
class PaletteColor:
    palette_index: int
    rgb: tuple[int, int, int]
    hex_color: str
    pixel_count: int
    is_background: bool = False


@dataclass(frozen=True)
class ColorRegionSpec:
    palette_index: int
    layer_id: str
    fill_color: str
    mask: np.ndarray
    scalar_field: np.ndarray
    pixel_count: int


@dataclass
class ColorSegmentationResult:
    palette: list[PaletteColor]
    label_image: np.ndarray
    background_index: int | None
    color_regions: list[ColorRegionSpec]


def segment_color_regions(
    preprocessed: PreprocessResult,
    config: HybridVectorizerConfig,
) -> ColorSegmentationResult:
    quantized = _quantize_image(preprocessed.color_image, config)
    labels = quantized["labels"]
    palette = quantized["palette"]
    if config.merge_similar_colors and len(palette) > 1:
        labels, palette = _merge_similar_palette_entries(
            labels,
            palette,
            valid_mask=preprocessed.valid_mask,
            distance_threshold=config.color_distance_threshold,
        )

    labels = np.where(preprocessed.valid_mask, labels, -1)
    background_index = _detect_background_index(labels, palette, preprocessed.valid_mask, config)
    palette_entries = _build_palette_entries(labels, palette, background_index)
    color_regions = _build_color_regions(preprocessed, labels, palette_entries, background_index)
    return ColorSegmentationResult(
        palette=palette_entries,
        label_image=labels,
        background_index=background_index,
        color_regions=color_regions,
    )


def _quantize_image(color_image: np.ndarray, config: HybridVectorizerConfig) -> dict[str, object]:
    image = Image.fromarray(np.asarray(color_image, dtype=np.uint8), mode="RGB")
    quantize_method = _quantize_method(config.quantization_method)
    dither_enum = getattr(getattr(Image, "Dither", Image), "NONE", 0)
    quantized = image.quantize(colors=max(1, int(config.max_colors)), method=quantize_method, dither=dither_enum)
    labels = np.asarray(quantized, dtype=np.int32)
    raw_palette = quantized.getpalette()[: 3 * max(1, labels.max(initial=0) + 1)]
    palette: list[tuple[int, int, int]] = []
    for index in range(max(1, labels.max(initial=0) + 1)):
        base = 3 * index
        if base + 2 >= len(raw_palette):
            palette.append((0, 0, 0))
            continue
        palette.append((int(raw_palette[base]), int(raw_palette[base + 1]), int(raw_palette[base + 2])))
    return {"labels": labels, "palette": palette}


def _quantize_method(name: str) -> int:
    normalized = name.strip().lower().replace("-", "_")
    quantize_enum = getattr(Image, "Quantize", None)
    mapping = {
        "median_cut": getattr(quantize_enum, "MEDIANCUT", 0) if quantize_enum is not None else 0,
        "max_coverage": getattr(quantize_enum, "MAXCOVERAGE", 1) if quantize_enum is not None else 1,
        "fast_octree": getattr(quantize_enum, "FASTOCTREE", 2) if quantize_enum is not None else 2,
    }
    return mapping.get(normalized, mapping["median_cut"])


def _merge_similar_palette_entries(
    labels: np.ndarray,
    palette: list[tuple[int, int, int]],
    valid_mask: np.ndarray,
    distance_threshold: float,
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    if len(palette) <= 1:
        return labels, palette

    counts = {index: int(np.sum((labels == index) & valid_mask)) for index in range(len(palette))}
    parents = list(range(len(palette)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parents[root_right] = root_left

    palette_array = np.asarray(palette, dtype=np.float32)
    for left in range(len(palette)):
        for right in range(left + 1, len(palette)):
            if np.linalg.norm(palette_array[left] - palette_array[right]) <= distance_threshold:
                union(left, right)

    cluster_members: dict[int, list[int]] = {}
    for index in range(len(palette)):
        root = find(index)
        cluster_members.setdefault(root, []).append(index)

    new_palette: list[tuple[int, int, int]] = []
    remap: dict[int, int] = {}
    for new_index, members in enumerate(
        sorted(cluster_members.values(), key=lambda group: sum(counts.get(idx, 0) for idx in group), reverse=True)
    ):
        total = max(1, sum(counts.get(idx, 0) for idx in members))
        weighted = sum(np.asarray(palette[idx], dtype=np.float32) * counts.get(idx, 0) for idx in members) / total
        rgb = tuple(int(round(float(value))) for value in weighted)
        new_palette.append(rgb)
        for old_index in members:
            remap[old_index] = new_index

    remapped_labels = np.full_like(labels, fill_value=-1)
    for old_index, new_index in remap.items():
        remapped_labels[labels == old_index] = new_index
    return remapped_labels, new_palette


def _build_palette_entries(
    labels: np.ndarray,
    palette: list[tuple[int, int, int]],
    background_index: int | None,
) -> list[PaletteColor]:
    entries: list[PaletteColor] = []
    for index, rgb in enumerate(palette):
        pixel_count = int(np.sum(labels == index))
        if pixel_count <= 0:
            continue
        entries.append(
            PaletteColor(
                palette_index=index,
                rgb=rgb,
                hex_color=_rgb_to_hex(rgb),
                pixel_count=pixel_count,
                is_background=index == background_index,
            )
        )
    entries.sort(key=lambda item: item.pixel_count, reverse=True)
    return entries


def _detect_background_index(
    labels: np.ndarray,
    palette: list[tuple[int, int, int]],
    valid_mask: np.ndarray,
    config: HybridVectorizerConfig,
) -> int | None:
    mode = config.background_mode.strip().lower()
    if mode == "none":
        return None
    if mode == "manual":
        if not config.background_color:
            return None
        rgb = _parse_hex_color(config.background_color)
        if rgb is None or not palette:
            return None
        distances = [np.linalg.norm(np.asarray(color, dtype=np.float32) - np.asarray(rgb, dtype=np.float32)) for color in palette]
        return int(np.argmin(distances))

    border = np.concatenate(
        [
            labels[0, :],
            labels[-1, :],
            labels[:, 0],
            labels[:, -1],
        ]
    )
    border = border[border >= 0]
    if border.size == 0:
        return None
    unique, counts = np.unique(border, return_counts=True)
    best = int(unique[np.argmax(counts)])
    support = float(np.max(counts)) / float(border.size)
    if support >= 0.35:
        return best

    if np.mean(valid_mask.astype(np.float32)) < 0.98:
        return None
    return best if support >= 0.22 else None


def _build_color_regions(
    preprocessed: PreprocessResult,
    labels: np.ndarray,
    palette_entries: list[PaletteColor],
    background_index: int | None,
) -> list[ColorRegionSpec]:
    if not palette_entries:
        return []

    palette_map = {entry.palette_index: np.asarray(entry.rgb, dtype=np.float32) for entry in palette_entries}
    active_indices = [entry.palette_index for entry in palette_entries if entry.palette_index != background_index]
    if not active_indices:
        return []

    rgb = preprocessed.color_image.astype(np.float32)
    region_specs: list[ColorRegionSpec] = []
    for order, entry in enumerate(sorted((item for item in palette_entries if not item.is_background), key=lambda item: item.pixel_count, reverse=True)):
        mask = labels == entry.palette_index
        if not np.any(mask):
            continue

        scalar_field = _membership_field(rgb, preprocessed.valid_mask, palette_map, entry.palette_index)
        region_specs.append(
            ColorRegionSpec(
                palette_index=entry.palette_index,
                layer_id=f"layer_{order:02d}_{entry.hex_color[1:]}",
                fill_color=entry.hex_color,
                mask=mask,
                scalar_field=scalar_field,
                pixel_count=entry.pixel_count,
            )
        )
    return region_specs


def _membership_field(
    rgb: np.ndarray,
    valid_mask: np.ndarray,
    palette_map: dict[int, np.ndarray],
    palette_index: int,
) -> np.ndarray:
    target = palette_map[palette_index]
    diff = rgb - target[None, None, :]
    target_distance = np.linalg.norm(diff, axis=2)

    other_distances = []
    for other_index, other_color in palette_map.items():
        if other_index == palette_index:
            continue
        other_distances.append(np.linalg.norm(rgb - other_color[None, None, :], axis=2))

    if other_distances:
        nearest_other = np.min(np.stack(other_distances, axis=0), axis=0)
        membership = nearest_other / np.maximum(target_distance + nearest_other, 1e-6)
    else:
        membership = 1.0 - (target_distance / MAX_COLOR_DISTANCE)

    membership = np.where(valid_mask, membership, 0.0)
    return membership.astype(np.float32)


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _parse_hex_color(text: str) -> tuple[int, int, int] | None:
    value = text.strip()
    if value.startswith("#"):
        value = value[1:]
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    if len(value) != 6:
        return None
    try:
        return tuple(int(value[index : index + 2], 16) for index in range(0, 6, 2))
    except ValueError:
        return None
