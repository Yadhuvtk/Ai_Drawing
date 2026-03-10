from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Generator


def ensure_dir(path: str) -> None:
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def sha1_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def jsonl_append(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def iter_svg_paths_from_manifest(manifest_path: str) -> Generator[str, None, None]:
    """
    Robust loader:
    - Each line can be JSON containing "path" or "svg_path"
    - Or a raw path string (fallback)
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # JSON line?
            if line.startswith("{") and line.endswith("}"):
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        if "path" in obj:
                            yield str(obj["path"])
                            continue
                        if "svg_path" in obj:
                            yield str(obj["svg_path"])
                            continue
                except Exception:
                    pass

            # Fallback: raw string path
            yield line