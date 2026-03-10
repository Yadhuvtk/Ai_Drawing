from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple

import yaml
from tqdm import tqdm

from yd_vector.data.im2svg_manifest import (
    ensure_dir,
    iter_svg_paths_from_manifest,
    jsonl_append,
    sha1_id,
)


@dataclass
class Config:
    raw_svg_manifest: str
    output_png_dir: str
    output_manifest: str
    failures_report: str
    image_size: int = 256
    limit: int = 10_000
    num_workers: int = 8
    overwrite: bool = False
    timeout_seconds: int = 30
    inkscape_exe: str = ""  # if empty: use env INKSCAPE_EXE or "inkscape"


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path: str) -> Config:
    d = load_yaml(path)

    # defaults
    cfg = Config(
        raw_svg_manifest=str(d.get("raw_svg_manifest", "data_local/manifest/manifest.jsonl")),
        output_png_dir=str(d.get("output_png_dir", "data_local/renders_256")),
        output_manifest=str(d.get("output_manifest", "data_local/manifest/im2svg_manifest_256.jsonl")),
        failures_report=str(d.get("failures_report", "data_local/manifest/im2svg_failures_256.jsonl")),
        image_size=int(d.get("image_size", 256)),
        limit=int(d.get("limit", 10_000)),
        num_workers=int(d.get("num_workers", 8)),
        overwrite=bool(d.get("overwrite", False)),
        timeout_seconds=int(d.get("timeout_seconds", 30)),
        inkscape_exe=str(d.get("inkscape_exe", "")).strip(),
    )

    # If num_workers == 0, auto
    if cfg.num_workers == 0:
        cfg.num_workers = os.cpu_count() or 8

    return cfg


def resolve_inkscape_exe(cfg: Config) -> str:
    # Priority: config > env INKSCAPE_EXE > "inkscape"
    exe = cfg.inkscape_exe or os.environ.get("INKSCAPE_EXE", "").strip() or "inkscape"
    return exe


def inkscape_cmd(inkscape_exe: str, svg_path: str, png_path: str, size: int) -> list[str]:
    # Use inkscape CLI to export png with background white
    # NOTE: inkscape.com is preferred for CLI on Windows.
    return [
        inkscape_exe,
        svg_path,
        "--export-type=png",
        f"--export-filename={png_path}",
        f"--export-width={size}",
        f"--export-height={size}",
        "--export-background=white",
    ]


def run_render_one(
    inkscape_exe: str,
    svg_path: str,
    png_path: str,
    size: int,
    overwrite: bool,
    timeout_seconds: int,
) -> Tuple[str, str, bool, Optional[str], Optional[int]]:
    """
    Returns: (id, svg_path, rendered_ok, error, returncode)
    """
    try:
        # skip if exists
        if (not overwrite) and os.path.exists(png_path) and os.path.getsize(png_path) > 0:
            return ("", svg_path, False, "SKIP_EXISTS", 0)

        ensure_dir(os.path.dirname(png_path))

        cmd = inkscape_cmd(inkscape_exe, svg_path, png_path, size)

        # IMPORTANT FIX:
        # Force UTF-8 decoding and replace invalid bytes so Windows cp1252 errors never crash.
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )

        if res.returncode != 0:
            err = (res.stderr or res.stdout or "").strip()
            if not err:
                err = "Inkscape returned non-zero with empty output"
            return ("", svg_path, False, err[:2000], res.returncode)

        # Ensure file was created
        if not os.path.exists(png_path) or os.path.getsize(png_path) == 0:
            return ("", svg_path, False, "PNG_NOT_CREATED_OR_EMPTY", res.returncode)

        return ("", svg_path, True, None, 0)

    except subprocess.TimeoutExpired:
        return ("", svg_path, False, f"TIMEOUT_{timeout_seconds}s", -1)
    except Exception as e:
        return ("", svg_path, False, f"{type(e).__name__}: {e}", -2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to configs/im2svg_data.yaml")
    p.add_argument("--limit", type=int, default=None, help="Override limit (use -1 for all)")
    p.add_argument("--size", type=int, default=None, help="Override image_size")
    p.add_argument("--workers", type=int, default=None, help="Override num_workers (0=auto)")
    p.add_argument("--fresh", action="store_true", help="Start new output manifest (delete output manifest only)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing PNGs")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    if args.limit is not None:
        cfg.limit = int(args.limit)
    if args.size is not None:
        cfg.image_size = int(args.size)
    if args.workers is not None:
        cfg.num_workers = int(args.workers)
        if cfg.num_workers == 0:
            cfg.num_workers = os.cpu_count() or 8
    if args.overwrite:
        cfg.overwrite = True

    inkscape_exe = resolve_inkscape_exe(cfg)

    print("Starting PNG<->SVG pair generation")
    print(f"raw_svg_manifest: {Path(cfg.raw_svg_manifest).resolve()}")
    print(f"output_png_dir:   {Path(cfg.output_png_dir).resolve()}")
    print(f"output_manifest:  {Path(cfg.output_manifest).resolve()}")
    print(f"failures_report:  {Path(cfg.failures_report).resolve()}")
    print(f"image_size:       {cfg.image_size}")
    print(f"limit:            {cfg.limit}")
    print(f"num_workers:      {cfg.num_workers}")
    print(f"overwrite:        {cfg.overwrite}")
    print(f"timeout_seconds:  {cfg.timeout_seconds}")
    print(f"inkscape_exe:     {inkscape_exe}")

    # sanity check inkscape
    try:
        ver = subprocess.run(
            [inkscape_exe, "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        if ver.returncode != 0:
            raise RuntimeError(ver.stderr.strip() or ver.stdout.strip() or "Inkscape version failed")
    except Exception:
        print("\n❌ Inkscape not found / not runnable.")
        print("Fix:")
        print("  1) Add Inkscape to PATH, OR")
        print("  2) Set env var INKSCAPE_EXE to inkscape.com path, e.g.:")
        print(r"     $env:INKSCAPE_EXE='C:\Program Files\Inkscape\bin\inkscape.com'")
        return 2

    ensure_dir(os.path.dirname(cfg.output_manifest))
    ensure_dir(cfg.output_png_dir)

    # --fresh deletes output manifest only (keeps PNGs)
    if args.fresh and os.path.exists(cfg.output_manifest):
        print(f"\n--fresh: deleting output manifest: {cfg.output_manifest}")
        os.remove(cfg.output_manifest)

    # Build a set of already-written IDs to make resume safe (optional, fast enough for 10k/100k)
    done_ids = set()
    if os.path.exists(cfg.output_manifest):
        try:
            with open(cfg.output_manifest, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if "id" in obj:
                        done_ids.add(obj["id"])
        except Exception:
            # if manifest is corrupted, user can pass --fresh
            pass

    total = cfg.limit if cfg.limit != -1 else None

    processed = 0
    rendered = 0
    skipped = 0
    failed = 0

    # Stream SVG paths
    svg_iter = iter_svg_paths_from_manifest(cfg.raw_svg_manifest)

    # Prepare tasks (we submit progressively to avoid huge memory)
    max_inflight = cfg.num_workers * 4

    def make_item(svg_path: str) -> Tuple[str, str, str]:
        svg_path_norm = os.path.normpath(svg_path)
        _id = sha1_id(svg_path_norm)
        png_path = os.path.join(cfg.output_png_dir, f"{_id}.png")
        return _id, svg_path_norm, png_path

    futures = []
    with ProcessPoolExecutor(max_workers=cfg.num_workers) as ex:
        pbar = tqdm(total=total, desc="Building pairs", unit="svg")

        def submit_one(_id: str, svg_path: str, png_path: str):
            return ex.submit(
                run_render_one,
                inkscape_exe,
                svg_path,
                png_path,
                cfg.image_size,
                cfg.overwrite,
                cfg.timeout_seconds,
            )

        # Prime inflight queue
        while len(futures) < max_inflight:
            try:
                svg_path = next(svg_iter)
            except StopIteration:
                break

            if cfg.limit != -1 and processed >= cfg.limit:
                break

            _id, svg_path, png_path = make_item(svg_path)

            processed += 1
            pbar.update(1)

            if _id in done_ids:
                skipped += 1
                continue

            if (not cfg.overwrite) and os.path.exists(png_path) and os.path.getsize(png_path) > 0:
                # already rendered on disk
                skipped += 1
                jsonl_append(cfg.output_manifest, {"id": _id, "png_path": png_path, "svg_path": svg_path, "size": cfg.image_size})
                done_ids.add(_id)
                continue

            futures.append((submit_one(_id, svg_path, png_path), _id, svg_path, png_path))

        # Main loop
        while futures:
            # Wait for one to complete
            done_fut, _id, svg_path, png_path = futures.pop(0)
            try:
                _, _, ok, err, rc = done_fut.result()
            except Exception as e:
                ok = False
                err = f"{type(e).__name__}: {e}"
                rc = -999

            if ok:
                rendered += 1
                jsonl_append(cfg.output_manifest, {"id": _id, "png_path": png_path, "svg_path": svg_path, "size": cfg.image_size})
                done_ids.add(_id)
            else:
                if err == "SKIP_EXISTS":
                    skipped += 1
                    jsonl_append(cfg.output_manifest, {"id": _id, "png_path": png_path, "svg_path": svg_path, "size": cfg.image_size})
                    done_ids.add(_id)
                else:
                    failed += 1
                    jsonl_append(cfg.failures_report, {"id": _id, "svg_path": svg_path, "png_path": png_path, "error": err, "returncode": rc})

            # Update tqdm postfix
            pbar.set_postfix(
                failed=failed,
                processed=processed,
                rendered=rendered,
                skipped=skipped,
            )

            # Submit next item to keep inflight
            while len(futures) < max_inflight:
                try:
                    svg_path_next = next(svg_iter)
                except StopIteration:
                    break

                if cfg.limit != -1 and processed >= cfg.limit:
                    break

                _id2, svg_path2, png_path2 = make_item(svg_path_next)

                processed += 1
                pbar.update(1)

                if _id2 in done_ids:
                    skipped += 1
                    continue

                if (not cfg.overwrite) and os.path.exists(png_path2) and os.path.getsize(png_path2) > 0:
                    skipped += 1
                    jsonl_append(cfg.output_manifest, {"id": _id2, "png_path": png_path2, "svg_path": svg_path2, "size": cfg.image_size})
                    done_ids.add(_id2)
                    continue

                futures.append((submit_one(_id2, svg_path2, png_path2), _id2, svg_path2, png_path2))

        pbar.close()

    print("\nDone.")
    print(f"Processed: {processed}")
    print(f"Rendered:  {rendered}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")
    print(f"Output manifest: {cfg.output_manifest}")
    print(f"Failures report: {cfg.failures_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())