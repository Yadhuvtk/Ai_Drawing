from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm

from yd_vector.data.svg_normalizer import normalize_svg_text
from yd_vector.data.svg_reader import read_svg_file
from yd_vector.paths import REPO_ROOT
from yd_vector.utils.io import append_jsonl, iter_svg_files, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize SVG files into a clean output directory.")
    parser.add_argument("--config", type=str, default="configs/data.yaml")
    parser.add_argument("--source_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    cfg = load_yaml(cfg_path)

    source_dir = Path(args.source_dir) if args.source_dir else Path(cfg.get("raw_svg_dir", "E:/Yadhu Projects/SVG"))
    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg.get("normalized_dir", "E:/Yadhu Projects/SVG_NORMALIZED"))
    output_dir.mkdir(parents=True, exist_ok=True)

    max_file_bytes = int(cfg.get("max_file_bytes", 5 * 1024 * 1024))
    min_chars = int(cfg.get("min_chars", 20))
    float_precision = cfg.get("float_precision", None)
    truncate_chars = int(cfg.get("truncate_chars", 200000))

    fail_report = REPO_ROOT / "data_local" / "manifest" / "normalize_failures.jsonl"
    if fail_report.exists():
        fail_report.unlink()

    total = 0
    written = 0
    skipped = 0
    for src_path in tqdm(iter_svg_files(source_dir), desc="Normalizing SVGs"):
        total += 1
        read_res = read_svg_file(src_path, max_file_bytes=max_file_bytes, min_chars=min_chars, truncate_chars=truncate_chars)
        if not read_res.ok:
            skipped += 1
            append_jsonl({"path": str(src_path), "stage": "read", "error": read_res.error}, fail_report)
            continue

        norm_res = normalize_svg_text(read_res.text, float_precision=float_precision)
        if not norm_res.ok:
            skipped += 1
            append_jsonl({"path": str(src_path), "stage": "normalize", "error": norm_res.error}, fail_report)
            continue

        dst_path = output_dir / src_path.name
        try:
            with open(dst_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(norm_res.text)
            written += 1
        except OSError as e:
            skipped += 1
            append_jsonl({"path": str(src_path), "stage": "write", "error": str(e)}, fail_report)

    print("Normalization complete")
    print(f"source_dir: {source_dir}")
    print(f"output_dir: {output_dir}")
    print(f"total: {total}")
    print(f"written: {written}")
    print(f"skipped: {skipped}")
    print(f"fail_report: {fail_report}")


if __name__ == "__main__":
    main()
