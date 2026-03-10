from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yd_vector.paths import REPO_ROOT
from yd_vector.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Export model artifacts from a training run.")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="latest", help="Checkpoint folder name under checkpoints/")
    args = parser.parse_args()

    run_dir = REPO_ROOT / "outputs" / "runs" / args.run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    ckpt_dir = run_dir / "checkpoints" / args.checkpoint
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint folder not found: {ckpt_dir}")

    export_dir = REPO_ROOT / "outputs" / "exports" / args.run_name
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    for name in ("model.pt", "optim.pt", "scaler.pt", "step.json"):
        src = ckpt_dir / name
        if src.exists():
            shutil.copy2(src, export_dir / name)

    tok_src = run_dir / "tokenizer.json"
    if tok_src.exists():
        shutil.copy2(tok_src, export_dir / "tokenizer.json")

    cfg_src = run_dir / "configs"
    if cfg_src.exists():
        shutil.copytree(cfg_src, export_dir / "configs")

    write_json(
        {
            "run_name": args.run_name,
            "checkpoint": args.checkpoint,
            "source_run_dir": str(run_dir),
            "exported_at": datetime.now().isoformat(timespec="seconds"),
        },
        export_dir / "export_meta.json",
        indent=2,
    )
    print(f"Export complete: {export_dir}")


if __name__ == "__main__":
    main()
