from __future__ import annotations

import argparse
import heapq
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm

from yd_vector.data.svg_reader import read_svg_file
from yd_vector.paths import REPO_ROOT
from yd_vector.utils.io import iter_svg_files, load_yaml, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan raw SVG dataset and report summary stats.")
    parser.add_argument("--config", type=str, default="configs/data.yaml")
    parser.add_argument("--top_n", type=int, default=10)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    cfg = load_yaml(cfg_path)

    raw_dir = Path(cfg.get("raw_svg_dir", "E:/Yadhu Projects/SVG"))
    max_file_bytes = int(cfg.get("max_file_bytes", 5 * 1024 * 1024))
    min_chars = int(cfg.get("min_chars", 20))
    truncate_chars = int(cfg.get("truncate_chars", 200000))

    if not raw_dir.exists():
        raise FileNotFoundError(f"raw_svg_dir not found: {raw_dir}")

    total_files = 0
    total_bytes = 0
    skipped_large = 0
    decode_failures = 0
    too_short = 0
    largest: list[tuple[int, str]] = []

    for path in tqdm(iter_svg_files(raw_dir), desc="Scanning SVGs"):
        total_files += 1
        size = path.stat().st_size
        total_bytes += size
        heapq.heappush(largest, (size, str(path)))
        if len(largest) > args.top_n:
            heapq.heappop(largest)

        result = read_svg_file(path, max_file_bytes=max_file_bytes, min_chars=min_chars, truncate_chars=truncate_chars)
        if not result.ok:
            if result.error == "file_too_large":
                skipped_large += 1
            elif result.error == "too_short":
                too_short += 1
            else:
                decode_failures += 1

    largest_sorted = sorted(largest, key=lambda x: x[0], reverse=True)
    summary = {
        "raw_dir": str(raw_dir),
        "total_svg_files": total_files,
        "total_bytes": total_bytes,
        "avg_bytes": (total_bytes / total_files) if total_files else 0.0,
        "skipped_large": skipped_large,
        "decode_failures": decode_failures,
        "too_short": too_short,
        "largest_files": [{"size_bytes": s, "path": p} for s, p in largest_sorted],
    }
    out_path = REPO_ROOT / "data_local" / "manifest" / "scan_summary.json"
    write_json(summary, out_path, indent=2)

    print("Scan complete")
    print(f"raw_dir: {summary['raw_dir']}")
    print(f"total_svg_files: {summary['total_svg_files']}")
    print(f"total_bytes: {summary['total_bytes']}")
    print(f"avg_bytes: {summary['avg_bytes']:.2f}")
    print(f"skipped_large: {summary['skipped_large']}")
    print(f"decode_failures: {summary['decode_failures']}")
    print(f"too_short: {summary['too_short']}")
    print(f"summary_json: {out_path}")


if __name__ == "__main__":
    main()
