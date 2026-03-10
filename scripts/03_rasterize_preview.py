from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yd_vector.data.rasterizer import has_cairosvg, rasterize_svg_to_png
from yd_vector.data.svg_reader import read_svg_file
from yd_vector.paths import REPO_ROOT
from yd_vector.utils.io import iter_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Optional SVG to PNG preview rasterization.")
    parser.add_argument("--manifest", type=str, default="data_local/manifest/manifest.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs/previews")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    args = parser.parse_args()

    if not has_cairosvg():
        print("cairosvg is not installed. Skipping preview rasterization.")
        return

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    done = 0
    for i, row in enumerate(iter_jsonl(manifest_path)):
        if done >= args.num_samples:
            break
        src = row.get("path")
        if not src:
            continue
        r = read_svg_file(src, max_file_bytes=20 * 1024 * 1024, min_chars=1, truncate_chars=500000)
        if not r.ok:
            continue
        out_png = output_dir / f"preview_{i:05d}.png"
        if rasterize_svg_to_png(r.text, out_png, width=args.width, height=args.height):
            done += 1

    print(f"Rasterization complete. Generated {done} previews in {output_dir}")


if __name__ == "__main__":
    main()
