from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from yd_vector.hybrid_vectorizer.config import HybridVectorizerConfig


@dataclass
class PreprocessResult:
    width: int
    height: int
    color_image: np.ndarray
    alpha: np.ndarray
    valid_mask: np.ndarray
    grayscale: np.ndarray
    foreground: np.ndarray
    binary_mask: np.ndarray
    threshold: int
    foreground_threshold: float


def preprocess_image(image_path: str | Path, config: HybridVectorizerConfig) -> PreprocessResult:
    rgba = Image.open(image_path).convert("RGBA")
    rgba = _resize_longest_side(rgba, config.target_size)

    alpha = np.asarray(rgba.getchannel("A"), dtype=np.float32) / 255.0
    image = _composite_over_white(rgba)
    color_image = image
    if config.color_preblur_radius > 0.0:
        color_image = color_image.filter(ImageFilter.GaussianBlur(radius=float(config.color_preblur_radius)))
    rgb = np.asarray(color_image, dtype=np.uint8)
    valid_mask = alpha > 1e-3
    grayscale_image = image.convert("L")
    if config.preblur_radius > 0.0:
        grayscale_image = grayscale_image.filter(ImageFilter.GaussianBlur(radius=float(config.preblur_radius)))

    grayscale = np.asarray(grayscale_image, dtype=np.uint8)
    threshold = int(config.threshold if config.threshold is not None else otsu_threshold(grayscale))
    threshold = max(0, min(255, threshold + int(config.threshold_bias)))

    foreground = _build_foreground_field(grayscale, alpha, config)
    foreground_threshold = _threshold_to_foreground_level(threshold, config.invert)
    binary_mask = foreground >= foreground_threshold

    if config.morph_open_iterations > 0:
        binary_mask = _binary_open(binary_mask, iterations=config.morph_open_iterations)
    if config.morph_close_iterations > 0:
        binary_mask = _binary_close(binary_mask, iterations=config.morph_close_iterations)
    binary_mask = _remove_small_components(binary_mask, min_area=max(2, config.min_region_area // 4))
    binary_mask = _fill_tiny_holes(
        binary_mask,
        max_area=max(1, config.min_hole_area // (4 if config.preserve_narrow_gaps else 2)),
    )

    height, width = grayscale.shape
    return PreprocessResult(
        width=width,
        height=height,
        color_image=rgb,
        alpha=alpha.astype(np.float32),
        valid_mask=valid_mask.astype(bool),
        grayscale=grayscale,
        foreground=foreground.astype(np.float32),
        binary_mask=binary_mask.astype(bool),
        threshold=threshold,
        foreground_threshold=float(foreground_threshold),
    )


def otsu_threshold(grayscale: np.ndarray) -> int:
    hist = np.bincount(grayscale.ravel(), minlength=256).astype(np.float64)
    total = float(grayscale.size)
    cumulative_weight = np.cumsum(hist)
    cumulative_mean = np.cumsum(hist * np.arange(256, dtype=np.float64))
    global_mean = cumulative_mean[-1]

    best_threshold = 127
    best_variance = -1.0
    for threshold in range(256):
        weight_background = cumulative_weight[threshold]
        weight_foreground = total - weight_background
        if weight_background == 0.0 or weight_foreground == 0.0:
            continue

        mean_background = cumulative_mean[threshold] / weight_background
        mean_foreground = (global_mean - cumulative_mean[threshold]) / weight_foreground
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if variance > best_variance:
            best_variance = variance
            best_threshold = threshold
    return best_threshold


def _composite_over_white(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


def _resize_longest_side(image: Image.Image, target_size: int | None) -> Image.Image:
    if target_size is None:
        return image

    width, height = image.size
    longest_side = max(width, height)
    if longest_side <= 0 or longest_side == target_size:
        return image

    scale = target_size / float(longest_side)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resampling = getattr(Image, "Resampling", Image)
    return image.resize(new_size, resampling.LANCZOS)


def _build_foreground_field(grayscale: np.ndarray, alpha: np.ndarray, config: HybridVectorizerConfig) -> np.ndarray:
    tone_foreground = 1.0 - (grayscale.astype(np.float32) / 255.0)
    if config.invert:
        tone_foreground = 1.0 - tone_foreground

    if not config.use_alpha_foreground:
        return tone_foreground

    if float(alpha.max()) <= 0.0 or float(alpha.min()) >= 0.999:
        return tone_foreground
    return np.maximum(tone_foreground, alpha.astype(np.float32))


def _threshold_to_foreground_level(threshold: int, invert: bool) -> float:
    if invert:
        return float(threshold) / 255.0
    return float(255 - threshold) / 255.0


def _binary_open(mask: np.ndarray, iterations: int) -> np.ndarray:
    current = np.asarray(mask, dtype=bool)
    for _ in range(max(0, iterations)):
        current = _binary_dilate(_binary_erode(current))
    return current


def _binary_close(mask: np.ndarray, iterations: int) -> np.ndarray:
    current = np.asarray(mask, dtype=bool)
    for _ in range(max(0, iterations)):
        current = _binary_erode(_binary_dilate(current))
    return current


def _binary_dilate(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    height, width = mask.shape
    neighborhoods = [
        padded[y : y + height, x : x + width]
        for y in range(3)
        for x in range(3)
    ]
    return np.logical_or.reduce(neighborhoods)


def _binary_erode(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    height, width = mask.shape
    neighborhoods = [
        padded[y : y + height, x : x + width]
        for y in range(3)
        for x in range(3)
    ]
    return np.logical_and.reduce(neighborhoods)


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return np.asarray(mask, dtype=bool)

    current = np.asarray(mask, dtype=bool)
    visited = np.zeros_like(current, dtype=bool)
    height, width = current.shape
    cleaned = current.copy()
    for y in range(height):
        for x in range(width):
            if not current[y, x] or visited[y, x]:
                continue
            component = _flood_fill(current, visited, y, x, target=True)
            if len(component) < min_area:
                for cy, cx in component:
                    cleaned[cy, cx] = False
    return cleaned


def _fill_tiny_holes(mask: np.ndarray, max_area: int) -> np.ndarray:
    if max_area <= 0:
        return np.asarray(mask, dtype=bool)

    inverse = ~np.asarray(mask, dtype=bool)
    visited = np.zeros_like(inverse, dtype=bool)
    height, width = inverse.shape
    filled = np.asarray(mask, dtype=bool).copy()
    for y in range(height):
        for x in range(width):
            if not inverse[y, x] or visited[y, x]:
                continue
            component = _flood_fill(inverse, visited, y, x, target=True)
            touches_border = any(cy in (0, height - 1) or cx in (0, width - 1) for cy, cx in component)
            if not touches_border and len(component) <= max_area:
                for cy, cx in component:
                    filled[cy, cx] = True
    return filled


def _flood_fill(mask: np.ndarray, visited: np.ndarray, start_y: int, start_x: int, target: bool) -> list[tuple[int, int]]:
    queue = [(start_y, start_x)]
    visited[start_y, start_x] = True
    points: list[tuple[int, int]] = []
    while queue:
        y, x = queue.pop()
        points.append((y, x))
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and mask[ny, nx] == target and not visited[ny, nx]:
                visited[ny, nx] = True
                queue.append((ny, nx))
    return points


# TODO: Add optional AI-assisted foreground extraction for difficult raster inputs.
