from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"
DATA_LOCAL_DIR = REPO_ROOT / "data_local"
MANIFEST_DIR = DATA_LOCAL_DIR / "manifest"
CACHE_DIR = DATA_LOCAL_DIR / "cache"
OUTPUTS_DIR = REPO_ROOT / "outputs"
RUNS_DIR = OUTPUTS_DIR / "runs"
EXPORTS_DIR = OUTPUTS_DIR / "exports"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
