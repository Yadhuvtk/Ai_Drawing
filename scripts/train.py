from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yd_vector.paths import REPO_ROOT
from yd_vector.training.trainer import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YD-Vector from scratch.")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--model_config", type=str, default=None)
    parser.add_argument("--data_config", type=str, default=None)
    parser.add_argument("--fresh", action="store_true", help="Ignore latest checkpoint and start fresh.")
    args = parser.parse_args()

    train_cfg = Path(args.config)
    if not train_cfg.is_absolute():
        train_cfg = REPO_ROOT / train_cfg

    model_cfg = Path(args.model_config) if args.model_config else None
    if model_cfg is not None and not model_cfg.is_absolute():
        model_cfg = REPO_ROOT / model_cfg

    data_cfg = Path(args.data_config) if args.data_config else None
    if data_cfg is not None and not data_cfg.is_absolute():
        data_cfg = REPO_ROOT / data_cfg

    run_training(train_config_path=train_cfg, model_config_path=model_cfg, data_config_path=data_cfg, fresh=args.fresh)


if __name__ == "__main__":
    main()
