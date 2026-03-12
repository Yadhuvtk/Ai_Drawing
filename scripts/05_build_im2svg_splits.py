from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yd_vector.paths import REPO_ROOT
from yd_vector.utils.io import iter_jsonl, write_json


def assign_split(key: str, seed: int, train_ratio: float, val_ratio: float) -> str:
    payload = f"{seed}:{key}".encode("utf-8")
    h = int(hashlib.sha1(payload).hexdigest(), 16)
    p = (h % 10_000_000) / 10_000_000.0
    if p < train_ratio:
        return "train"
    if p < train_ratio + val_ratio:
        return "val"
    return "test"


def row_id(row: dict) -> str:
    if row.get("id"):
        return str(row["id"])
    if row.get("svg_path"):
        return hashlib.sha1(str(row["svg_path"]).encode("utf-8")).hexdigest()
    if row.get("png_path"):
        return hashlib.sha1(str(row["png_path"]).encode("utf-8")).hexdigest()
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic train/val/test splits for an IM2SVG manifest.")
    parser.add_argument("--manifest", type=str, default="data_local/manifest/im2svg_manifest_256.jsonl")
    parser.add_argument("--out", type=str, default="data_local/manifest/im2svg_splits.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.95)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.0)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    if not manifest_path.exists():
        raise FileNotFoundError(f"IM2SVG manifest not found: {manifest_path}")

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path

    splits = {"train": [], "val": [], "test": []}
    seen_ids: set[str] = set()
    skipped = 0

    for row in iter_jsonl(manifest_path):
        item_id = row_id(row)
        if not item_id or item_id in seen_ids:
            skipped += 1
            continue
        split = assign_split(item_id, args.seed, args.train_ratio, args.val_ratio)
        splits[split].append(item_id)
        seen_ids.add(item_id)

    counts = {split: len(ids) for split, ids in splits.items()}
    total_written = sum(counts.values())
    payload = {
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "counts": counts,
        "manifest_path": str(manifest_path),
        "written": total_written,
        "skipped": skipped,
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
    }
    write_json(payload, out_path, indent=2)

    print("IM2SVG split build complete")
    print(f"manifest_path: {manifest_path}")
    print(f"out_path: {out_path}")
    print(f"written: {total_written}")
    print(f"skipped: {skipped}")
    print(f"counts: {counts}")


if __name__ == "__main__":
    main()
