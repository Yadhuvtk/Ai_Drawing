from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yd_vector.paths import REPO_ROOT, ensure_dir
from yd_vector.utils.io import write_json

CANVAS_SIZE = 128.0
CANVAS_CENTER = CANVAS_SIZE / 2.0
FILL_COLOR = "#000000"

LETTER_PATTERNS = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01110"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "J": ["00001", "00001", "00001", "00001", "10001", "10001", "01110"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01110", "10001", "10000", "01110", "00001", "10001", "01110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
}


@dataclass(frozen=True)
class PathSpec:
    d: str
    fill_rule: str = "nonzero"


def fmt(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text if text else "0"


def rect_path(x: float, y: float, w: float, h: float) -> str:
    return (
        f"M {fmt(x)} {fmt(y)} "
        f"H {fmt(x + w)} "
        f"V {fmt(y + h)} "
        f"H {fmt(x)} Z"
    )


def rounded_rect_path(x: float, y: float, w: float, h: float, radius: float) -> str:
    r = max(0.0, min(radius, w / 2.0, h / 2.0))
    if r == 0:
        return rect_path(x, y, w, h)
    return (
        f"M {fmt(x + r)} {fmt(y)} "
        f"H {fmt(x + w - r)} "
        f"A {fmt(r)} {fmt(r)} 0 0 1 {fmt(x + w)} {fmt(y + r)} "
        f"V {fmt(y + h - r)} "
        f"A {fmt(r)} {fmt(r)} 0 0 1 {fmt(x + w - r)} {fmt(y + h)} "
        f"H {fmt(x + r)} "
        f"A {fmt(r)} {fmt(r)} 0 0 1 {fmt(x)} {fmt(y + h - r)} "
        f"V {fmt(y + r)} "
        f"A {fmt(r)} {fmt(r)} 0 0 1 {fmt(x + r)} {fmt(y)} Z"
    )


def polygon_path(points: list[tuple[float, float]]) -> str:
    start_x, start_y = points[0]
    segments = [f"M {fmt(start_x)} {fmt(start_y)}"]
    for x, y in points[1:]:
        segments.append(f"L {fmt(x)} {fmt(y)}")
    segments.append("Z")
    return " ".join(segments)


def circle_path(cx: float, cy: float, radius: float) -> str:
    return (
        f"M {fmt(cx + radius)} {fmt(cy)} "
        f"A {fmt(radius)} {fmt(radius)} 0 1 0 {fmt(cx - radius)} {fmt(cy)} "
        f"A {fmt(radius)} {fmt(radius)} 0 1 0 {fmt(cx + radius)} {fmt(cy)} Z"
    )


def regular_polygon_points(
    sides: int,
    radius: float,
    rotation_deg: float = -90.0,
    cx: float = CANVAS_CENTER,
    cy: float = CANVAS_CENTER,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for i in range(sides):
        angle = math.radians(rotation_deg + (360.0 * i / sides))
        points.append((cx + math.cos(angle) * radius, cy + math.sin(angle) * radius))
    return points


def star_points(
    outer_radius: float,
    inner_radius: float,
    points: int = 5,
    rotation_deg: float = -90.0,
    cx: float = CANVAS_CENTER,
    cy: float = CANVAS_CENTER,
) -> list[tuple[float, float]]:
    vertices: list[tuple[float, float]] = []
    step = 360.0 / (points * 2)
    for i in range(points * 2):
        radius = outer_radius if i % 2 == 0 else inner_radius
        angle = math.radians(rotation_deg + step * i)
        vertices.append((cx + math.cos(angle) * radius, cy + math.sin(angle) * radius))
    return vertices


def path_svg(path_specs: list[PathSpec], size: int) -> str:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" width="{size}" height="{size}">',
    ]
    for spec in path_specs:
        if spec.fill_rule == "nonzero":
            lines.append(f'  <path fill="{FILL_COLOR}" d="{spec.d}"/>')
        else:
            lines.append(f'  <path fill="{FILL_COLOR}" fill-rule="{spec.fill_rule}" d="{spec.d}"/>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def make_ring_spec(outer_d: str, inner_d: str) -> PathSpec:
    return PathSpec(d=f"{outer_d} {inner_d}", fill_rule="evenodd")


def letter_path(pattern: list[str], cell_w: float, cell_h: float, slant_per_row: float = 0.0) -> str:
    rows = len(pattern)
    cols = len(pattern[0])
    base_w = cols * cell_w
    base_h = rows * cell_h
    max_slant = max(0.0, slant_per_row * (rows - 1))
    start_x = (CANVAS_SIZE - base_w - max_slant) / 2.0
    start_y = (CANVAS_SIZE - base_h) / 2.0

    segments: list[str] = []
    for row_index, row in enumerate(pattern):
        offset_x = slant_per_row * row_index
        for col_index, value in enumerate(row):
            if value != "1":
                continue
            x = start_x + offset_x + col_index * cell_w
            y = start_y + row_index * cell_h
            segments.append(rect_path(x, y, cell_w, cell_h))
    return " ".join(segments)


def centered_square(size: float) -> str:
    x = CANVAS_CENTER - (size / 2.0)
    return rect_path(x, x, size, size)


def centered_square_ring(outer: float, inner: float) -> PathSpec:
    outer_x = CANVAS_CENTER - (outer / 2.0)
    inner_x = CANVAS_CENTER - (inner / 2.0)
    return make_ring_spec(rect_path(outer_x, outer_x, outer, outer), rect_path(inner_x, inner_x, inner, inner))


def centered_circle(radius: float) -> str:
    return circle_path(CANVAS_CENTER, CANVAS_CENTER, radius)


def centered_circle_ring(outer: float, inner: float) -> PathSpec:
    return make_ring_spec(centered_circle(outer), centered_circle(inner))


def centered_triangle(radius: float) -> str:
    return polygon_path(regular_polygon_points(3, radius))


def centered_triangle_ring(outer: float, inner: float) -> PathSpec:
    outer_path = polygon_path(regular_polygon_points(3, outer))
    inner_path = polygon_path(regular_polygon_points(3, inner))
    return make_ring_spec(outer_path, inner_path)


def centered_diamond(radius: float) -> str:
    return polygon_path(regular_polygon_points(4, radius, rotation_deg=-45.0))


def centered_diamond_ring(outer: float, inner: float) -> PathSpec:
    outer_path = polygon_path(regular_polygon_points(4, outer, rotation_deg=-45.0))
    inner_path = polygon_path(regular_polygon_points(4, inner, rotation_deg=-45.0))
    return make_ring_spec(outer_path, inner_path)


def centered_star(outer: float, inner: float) -> str:
    return polygon_path(star_points(outer, inner))


def centered_cross(span: float, arm: float) -> str:
    half_span = span / 2.0
    half_arm = arm / 2.0
    vertical = rect_path(CANVAS_CENTER - half_arm, CANVAS_CENTER - half_span, arm, span)
    horizontal = rect_path(CANVAS_CENTER - half_span, CANVAS_CENTER - half_arm, span, arm)
    return f"{vertical} {horizontal}"


def centered_rounded_square(size: float, radius: float) -> str:
    x = CANVAS_CENTER - (size / 2.0)
    return rounded_rect_path(x, x, size, size, radius)


def centered_rounded_square_ring(outer: float, outer_radius: float, inner: float, inner_radius: float) -> PathSpec:
    outer_x = CANVAS_CENTER - (outer / 2.0)
    inner_x = CANVAS_CENTER - (inner / 2.0)
    outer_path = rounded_rect_path(outer_x, outer_x, outer, outer, outer_radius)
    inner_path = rounded_rect_path(inner_x, inner_x, inner, inner, inner_radius)
    return make_ring_spec(outer_path, inner_path)


def house_icon(width: float, roof_height: float, body_height: float) -> str:
    body_w = width * 0.7
    body_x = CANVAS_CENTER - (body_w / 2.0)
    body_y = CANVAS_CENTER - (body_height * 0.15)
    roof_y = body_y - roof_height
    roof = polygon_path(
        [
            (CANVAS_CENTER, roof_y),
            (CANVAS_CENTER + width / 2.0, body_y),
            (CANVAS_CENTER - width / 2.0, body_y),
        ]
    )
    body = rect_path(body_x, body_y, body_w, body_height)
    return f"{roof} {body}"


def arrow_right_icon(length: float, thickness: float, head: float) -> str:
    shaft_x = CANVAS_CENTER - (length / 2.0)
    shaft_y = CANVAS_CENTER - (thickness / 2.0)
    shaft = rect_path(shaft_x, shaft_y, length - head, thickness)
    tip_x = shaft_x + length
    head_shape = polygon_path(
        [
            (shaft_x + length - head, CANVAS_CENTER - head / 2.0),
            (tip_x, CANVAS_CENTER),
            (shaft_x + length - head, CANVAS_CENTER + head / 2.0),
        ]
    )
    return f"{shaft} {head_shape}"


def chat_bubble_icon(width: float, height: float, radius: float, tail_w: float, tail_h: float) -> str:
    x = CANVAS_CENTER - (width / 2.0)
    y = CANVAS_CENTER - (height / 2.0) - 6.0
    bubble = rounded_rect_path(x, y, width, height, radius)
    tail = polygon_path(
        [
            (CANVAS_CENTER - tail_w * 0.2, y + height),
            (CANVAS_CENTER + tail_w * 0.4, y + height),
            (CANVAS_CENTER - tail_w * 0.8, y + height + tail_h),
        ]
    )
    return f"{bubble} {tail}"


def flag_icon(pole_h: float, pole_w: float, flag_w: float, flag_h: float) -> str:
    pole_x = CANVAS_CENTER - (flag_w / 2.0)
    pole_y = CANVAS_CENTER - (pole_h / 2.0)
    pole = rect_path(pole_x, pole_y, pole_w, pole_h)
    flag = polygon_path(
        [
            (pole_x + pole_w, pole_y + 4.0),
            (pole_x + pole_w + flag_w, pole_y + 10.0),
            (pole_x + pole_w + flag_w * 0.72, pole_y + 10.0 + flag_h * 0.58),
            (pole_x + pole_w, pole_y + flag_h),
        ]
    )
    return f"{pole} {flag}"


def lightning_icon(width: float, height: float) -> str:
    x = CANVAS_CENTER - (width / 2.0)
    y = CANVAS_CENTER - (height / 2.0)
    return polygon_path(
        [
            (x + width * 0.58, y),
            (x + width * 0.22, y + height * 0.48),
            (x + width * 0.48, y + height * 0.48),
            (x + width * 0.34, y + height),
            (x + width * 0.82, y + height * 0.42),
            (x + width * 0.56, y + height * 0.42),
        ]
    )


def pin_icon(outer_r: float, inner_r: float, tip_y: float) -> PathSpec:
    top_circle = circle_path(CANVAS_CENTER, 50.0, outer_r)
    tail = polygon_path(
        [
            (CANVAS_CENTER - outer_r * 0.76, 58.0),
            (CANVAS_CENTER + outer_r * 0.76, 58.0),
            (CANVAS_CENTER, tip_y),
        ]
    )
    hole = circle_path(CANVAS_CENTER, 50.0, inner_r)
    return make_ring_spec(f"{top_circle} {tail}", hole)


def generate_letter_assets() -> dict[str, str]:
    variant_specs = {
        "regular": {"cell_w": 12.0, "cell_h": 12.0, "slant": 0.0},
        "wide": {"cell_w": 14.0, "cell_h": 12.0, "slant": 0.0},
        "tall": {"cell_w": 12.0, "cell_h": 14.0, "slant": 0.0},
        "slanted": {"cell_w": 11.5, "cell_h": 12.0, "slant": 2.0},
    }
    assets: dict[str, str] = {}
    for letter, pattern in LETTER_PATTERNS.items():
        for variant, spec in variant_specs.items():
            name = f"letter_{letter}_{variant}"
            path = letter_path(pattern, spec["cell_w"], spec["cell_h"], spec["slant"])
            assets[name] = path_svg([PathSpec(path)], size=int(CANVAS_SIZE))
    return assets


def generate_shape_assets() -> dict[str, str]:
    assets: dict[str, str] = {}
    specs: list[tuple[str, list[PathSpec]]] = [
        ("shape_square_fill_small", [PathSpec(centered_square(52.0))]),
        ("shape_square_fill_medium", [PathSpec(centered_square(64.0))]),
        ("shape_square_fill_large", [PathSpec(centered_square(78.0))]),
        ("shape_square_ring_medium", [centered_square_ring(78.0, 46.0)]),
        ("shape_circle_fill_small", [PathSpec(centered_circle(24.0))]),
        ("shape_circle_fill_medium", [PathSpec(centered_circle(32.0))]),
        ("shape_circle_fill_large", [PathSpec(centered_circle(40.0))]),
        ("shape_circle_ring_medium", [centered_circle_ring(40.0, 24.0)]),
        ("shape_triangle_fill_small", [PathSpec(centered_triangle(31.0))]),
        ("shape_triangle_fill_medium", [PathSpec(centered_triangle(38.0))]),
        ("shape_triangle_fill_large", [PathSpec(centered_triangle(44.0))]),
        ("shape_triangle_ring_medium", [centered_triangle_ring(44.0, 24.0)]),
        ("shape_diamond_fill_small", [PathSpec(centered_diamond(30.0))]),
        ("shape_diamond_fill_medium", [PathSpec(centered_diamond(38.0))]),
        ("shape_diamond_fill_large", [PathSpec(centered_diamond(46.0))]),
        ("shape_diamond_ring_medium", [centered_diamond_ring(46.0, 26.0)]),
        ("shape_star_fill_small", [PathSpec(centered_star(28.0, 12.0))]),
        ("shape_star_fill_medium", [PathSpec(centered_star(36.0, 16.0))]),
        ("shape_star_fill_large", [PathSpec(centered_star(42.0, 18.0))]),
        ("shape_star_fill_wide", [PathSpec(centered_star(40.0, 20.0))]),
        ("shape_cross_fill_small", [PathSpec(centered_cross(58.0, 16.0))]),
        ("shape_cross_fill_medium", [PathSpec(centered_cross(72.0, 18.0))]),
        ("shape_cross_fill_large", [PathSpec(centered_cross(84.0, 22.0))]),
        ("shape_cross_fill_wide", [PathSpec(centered_cross(76.0, 28.0))]),
        ("shape_rounded_square_fill_small", [PathSpec(centered_rounded_square(56.0, 10.0))]),
        ("shape_rounded_square_fill_medium", [PathSpec(centered_rounded_square(68.0, 12.0))]),
        ("shape_rounded_square_fill_large", [PathSpec(centered_rounded_square(82.0, 16.0))]),
        (
            "shape_rounded_square_ring_medium",
            [centered_rounded_square_ring(82.0, 16.0, 50.0, 8.0)],
        ),
    ]
    for name, path_specs in specs:
        assets[name] = path_svg(path_specs, size=int(CANVAS_SIZE))
    return assets


def generate_icon_assets() -> dict[str, str]:
    assets: dict[str, str] = {}
    specs: list[tuple[str, list[PathSpec]]] = [
        ("icon_house_compact", [PathSpec(house_icon(56.0, 28.0, 36.0))]),
        ("icon_house_medium", [PathSpec(house_icon(66.0, 32.0, 40.0))]),
        ("icon_house_wide", [PathSpec(house_icon(76.0, 30.0, 38.0))]),
        ("icon_house_tall", [PathSpec(house_icon(60.0, 36.0, 46.0))]),
        ("icon_arrow_right_compact", [PathSpec(arrow_right_icon(60.0, 16.0, 22.0))]),
        ("icon_arrow_right_medium", [PathSpec(arrow_right_icon(74.0, 18.0, 26.0))]),
        ("icon_arrow_right_wide", [PathSpec(arrow_right_icon(86.0, 18.0, 30.0))]),
        ("icon_arrow_right_thick", [PathSpec(arrow_right_icon(74.0, 26.0, 28.0))]),
        ("icon_chat_bubble_compact", [PathSpec(chat_bubble_icon(54.0, 38.0, 10.0, 14.0, 12.0))]),
        ("icon_chat_bubble_medium", [PathSpec(chat_bubble_icon(66.0, 44.0, 12.0, 16.0, 14.0))]),
        ("icon_chat_bubble_wide", [PathSpec(chat_bubble_icon(78.0, 42.0, 12.0, 18.0, 14.0))]),
        ("icon_chat_bubble_tall", [PathSpec(chat_bubble_icon(62.0, 52.0, 12.0, 16.0, 16.0))]),
        ("icon_flag_compact", [PathSpec(flag_icon(58.0, 8.0, 28.0, 24.0))]),
        ("icon_flag_medium", [PathSpec(flag_icon(70.0, 8.0, 34.0, 28.0))]),
        ("icon_flag_wide", [PathSpec(flag_icon(70.0, 8.0, 42.0, 28.0))]),
        ("icon_flag_tall", [PathSpec(flag_icon(82.0, 8.0, 34.0, 32.0))]),
        ("icon_lightning_compact", [PathSpec(lightning_icon(42.0, 64.0))]),
        ("icon_lightning_medium", [PathSpec(lightning_icon(50.0, 76.0))]),
        ("icon_lightning_wide", [PathSpec(lightning_icon(58.0, 76.0))]),
        ("icon_lightning_tall", [PathSpec(lightning_icon(48.0, 88.0))]),
        ("icon_pin_compact", [pin_icon(18.0, 8.0, 94.0)]),
        ("icon_pin_medium", [pin_icon(22.0, 10.0, 100.0)]),
        ("icon_pin_wide", [pin_icon(24.0, 11.0, 100.0)]),
        ("icon_pin_tall", [pin_icon(21.0, 9.0, 108.0)]),
    ]
    for name, path_specs in specs:
        assets[name] = path_svg(path_specs, size=int(CANVAS_SIZE))
    return assets


def generate_assets() -> dict[str, str]:
    assets: dict[str, str] = {}
    assets.update(generate_letter_assets())
    assets.update(generate_shape_assets())
    assets.update(generate_icon_assets())
    return dict(sorted(assets.items()))


def build_manifest(asset_names: list[str], out_dir: Path) -> dict:
    groups = {"letters": [], "shapes": [], "icons": []}
    for name in asset_names:
        if name.startswith("letter_"):
            groups["letters"].append(name)
        elif name.startswith("shape_"):
            groups["shapes"].append(name)
        else:
            groups["icons"].append(name)
    return {
        "canvas_size": int(CANVAS_SIZE),
        "fill": FILL_COLOR,
        "output_dir": str(out_dir),
        "counts": {key: len(value) for key, value in groups.items()},
        "total": len(asset_names),
        "files": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate simple stage-1 SVG curriculum assets.")
    parser.add_argument("--out_dir", type=str, default="data_local/curriculum_stage1/svg")
    parser.add_argument("--manifest", type=str, default="data_local/curriculum_stage1/asset_manifest.json")
    parser.add_argument("--clean", action="store_true", help="Delete existing SVG files in the output directory first.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    ensure_dir(out_dir)

    if args.clean:
        for path in out_dir.glob("*.svg"):
            path.unlink()

    assets = generate_assets()
    for name, svg_text in assets.items():
        (out_dir / f"{name}.svg").write_text(svg_text, encoding="utf-8", newline="\n")

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    write_json(build_manifest(list(assets.keys()), out_dir), manifest_path, indent=2)

    print("Stage-1 curriculum SVG generation complete")
    print(f"output_dir: {out_dir}")
    print(f"manifest_path: {manifest_path}")
    print(f"total_svg_files: {len(assets)}")


if __name__ == "__main__":
    main()
