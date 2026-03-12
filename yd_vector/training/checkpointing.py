from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Iterable

import torch

from yd_vector.utils.io import read_json, write_json


def save_checkpoint(
    run_dir: str | Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None = None,
    epoch: int = 0,
    best_val: float | None = None,
) -> Path:
    run_dir = Path(run_dir)
    ckpt_root = run_dir / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    step_dir = ckpt_root / f"step_{step:08d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), step_dir / "model.pt")
    torch.save(optimizer.state_dict(), step_dir / "optim.pt")
    if scaler is not None:
        torch.save(scaler.state_dict(), step_dir / "scaler.pt")

    write_json({"step": step, "epoch": epoch, "best_val": best_val}, step_dir / "step.json", indent=2)

    tokenizer_file = run_dir / "tokenizer.json"
    if tokenizer_file.exists():
        shutil.copy2(tokenizer_file, step_dir / "tokenizer.json")
    cfg_dir = run_dir / "configs"
    if cfg_dir.exists():
        cfg_dst = step_dir / "configs"
        if cfg_dst.exists():
            shutil.rmtree(cfg_dst)
        shutil.copytree(cfg_dir, cfg_dst)

    latest = ckpt_root / "latest"
    if latest.exists():
        shutil.rmtree(latest)
    shutil.copytree(step_dir, latest)
    return step_dir


def load_latest_checkpoint(
    run_dir: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any] | None:
    run_dir = Path(run_dir)
    latest = run_dir / "checkpoints" / "latest"
    if not (latest / "model.pt").exists():
        return None

    model.load_state_dict(torch.load(latest / "model.pt", map_location=map_location))
    if optimizer is not None and (latest / "optim.pt").exists():
        optimizer.load_state_dict(torch.load(latest / "optim.pt", map_location=map_location))
    if scaler is not None and (latest / "scaler.pt").exists():
        scaler.load_state_dict(torch.load(latest / "scaler.pt", map_location=map_location))

    state = {"step": 0, "epoch": 0, "best_val": None}
    if (latest / "step.json").exists():
        state.update(read_json(latest / "step.json"))
    return state


def load_checkpoint_weights(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    map_location: str | torch.device = "cpu",
    ignore_prefixes: Iterable[str] | None = None,
) -> Dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    raw_state = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(raw_state, dict) and "model_state_dict" in raw_state:
        source_state = raw_state["model_state_dict"]
    else:
        source_state = raw_state
    if not isinstance(source_state, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    prefixes = tuple(str(prefix) for prefix in (ignore_prefixes or []))
    target_state = model.state_dict()
    compatible_state: Dict[str, torch.Tensor] = {}
    ignored_keys: list[str] = []
    unexpected_keys: list[str] = []
    shape_mismatch_keys: list[str] = []

    for key, value in source_state.items():
        if prefixes and any(key.startswith(prefix) for prefix in prefixes):
            ignored_keys.append(key)
            continue
        if key not in target_state:
            unexpected_keys.append(key)
            continue
        if target_state[key].shape != value.shape:
            shape_mismatch_keys.append(key)
            continue
        compatible_state[key] = value

    model.load_state_dict(compatible_state, strict=False)
    missing_keys = [key for key in target_state.keys() if key not in compatible_state]
    return {
        "loaded_keys": sorted(compatible_state.keys()),
        "loaded_count": len(compatible_state),
        "missing_keys": sorted(missing_keys),
        "missing_count": len(missing_keys),
        "ignored_keys": sorted(ignored_keys),
        "ignored_count": len(ignored_keys),
        "unexpected_keys": sorted(unexpected_keys),
        "unexpected_count": len(unexpected_keys),
        "shape_mismatch_keys": sorted(shape_mismatch_keys),
        "shape_mismatch_count": len(shape_mismatch_keys),
    }
