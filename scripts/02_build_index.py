from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm

from yd_vector.data.svg_reader import read_svg_file
from yd_vector.paths import REPO_ROOT
from yd_vector.utils.io import iter_svg_files, load_yaml, write_json


def assign_split(path: str, seed: int, train_ratio: float, val_ratio: float) -> str:
    key = f"{seed}:{path}".encode("utf-8")
    h = int(hashlib.sha1(key).hexdigest(), 16)
    p = (h % 10_000_000) / 10_000_000.0
    if p < train_ratio:
        return "train"
    if p < train_ratio + val_ratio:
        return "val"
    return "test"


def choose_source_dir(raw_dir: Path, normalized_dir: Path, mode: str) -> tuple[Path, bool]:
    mode = mode.lower()
    if mode == "raw":
        return raw_dir, False
    if mode == "normalized":
        return normalized_dir, True
    if normalized_dir.exists():
        for p in normalized_dir.iterdir():
            if p.is_file() and p.suffix.lower() == ".svg":
                return normalized_dir, True
    return raw_dir, False


def main() -> None:
    parser = argparse.ArgumentParser(description="Build JSONL manifest and deterministic splits metadata.")
    parser.add_argument("--config", type=str, default="configs/data.yaml")
    parser.add_argument("--out_dir", type=str, default="data_local/manifest")
    parser.add_argument("--source", type=str, default="auto", choices=["auto", "raw", "normalized"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.98)
    parser.add_argument("--val_ratio", type=float, default=0.01)
    parser.add_argument("--test_ratio", type=float, default=0.01)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    cfg = load_yaml(cfg_path)

    raw_dir = Path(cfg.get("raw_svg_dir", "E:/Yadhu Projects/SVG"))
    normalized_dir = Path(cfg.get("normalized_dir", "E:/Yadhu Projects/SVG_NORMALIZED"))
    max_file_bytes = int(cfg.get("max_file_bytes", 5 * 1024 * 1024))
    min_chars = int(cfg.get("min_chars", 20))
    truncate_chars = int(cfg.get("truncate_chars", 200000))

    source_dir, using_normalized = choose_source_dir(raw_dir, normalized_dir, args.source)
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.jsonl"
    splits_path = out_dir / "splits.json"
    if manifest_path.exists():
        manifest_path.unlink()

    counts = {"train": 0, "val": 0, "test": 0}
    skipped = 0
    total = 0
    written = 0
    bytes_total = 0

    with open(manifest_path, "w", encoding="utf-8") as f:
        for svg_path in tqdm(iter_svg_files(source_dir), desc="Building index"):
            total += 1
            r = read_svg_file(svg_path, max_file_bytes=max_file_bytes, min_chars=min_chars, truncate_chars=truncate_chars)
            if not r.ok:
                skipped += 1
                continue
            path_str = str(svg_path.resolve())
            split = assign_split(path_str, args.seed, args.train_ratio, args.val_ratio)
            lt = r.text.lower()
            has_svg_tags = "<svg" in lt and "</svg>" in lt
            if not has_svg_tags:
                skipped += 1
                continue

            row = {
                "path": path_str,
                "size_bytes": int(r.size_bytes),
                "chars": len(r.text),
                "split": split,
                "has_svg_tags": has_svg_tags,
                "normalized": using_normalized,
            }
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
            written += 1
            bytes_total += int(r.size_bytes)
            counts[split] += 1

    split_meta = {
        "seed": args.seed,
        "ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio},
        "counts": counts,
        "source_dir": str(source_dir),
        "using_normalized": using_normalized,
        "manifest_path": str(manifest_path),
        "written": written,
        "skipped": skipped,
        "total_seen": total,
        "total_bytes": bytes_total,
    }
    write_json(split_meta, splits_path, indent=2)

    print("Index build complete")
    print(f"source_dir: {source_dir}")
    print(f"manifest_path: {manifest_path}")
    print(f"splits_path: {splits_path}")
    print(f"total_seen: {total}")
    print(f"written: {written}")
    print(f"skipped: {skipped}")
    print(f"counts: {counts}")


if __name__ == "__main__":
    main()
