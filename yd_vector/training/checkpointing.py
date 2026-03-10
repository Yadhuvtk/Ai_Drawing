from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict

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
