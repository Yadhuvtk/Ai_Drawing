from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from yd_vector.data.collate import causal_lm_collate
from yd_vector.data.dataset import SVGIterableDataset, count_split_records
from yd_vector.data.im2svg_collate import im2svg_collate_fn
from yd_vector.data.im2svg_dataset import IM2SVGDataset
from yd_vector.data.svg_tokenizer import build_tokenizer
from yd_vector.model.config import ModelConfig
from yd_vector.model.yd_vector_arch import YDVectorForCausalLM
from yd_vector.paths import REPO_ROOT, ensure_dir
from yd_vector.training.amp import autocast_context, build_scaler
from yd_vector.training.checkpointing import load_checkpoint_weights, load_latest_checkpoint, save_checkpoint
from yd_vector.training.eval import evaluate
from yd_vector.training.optim import build_optimizer, build_scheduler
from yd_vector.utils.io import dump_yaml, load_yaml, read_json
from yd_vector.utils.logging import setup_logger
from yd_vector.utils.metrics import loss_to_perplexity
from yd_vector.utils.seed import set_seed


def _resolve_cfg_paths(train_cfg_path: str | Path, model_cfg_path: str | Path | None, data_cfg_path: str | Path | None):
    train_cfg_path = Path(train_cfg_path)
    cfg_dir = train_cfg_path.parent
    if model_cfg_path is None:
        model_cfg_path = cfg_dir / "model.yaml"
    if data_cfg_path is None:
        data_cfg_path = cfg_dir / "data.yaml"
    return Path(model_cfg_path), Path(data_cfg_path)


def _resolve_repo_path(path_value: str | Path | None) -> Path | None:
    if path_value in (None, ""):
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _use_im2svg(train_cfg: Dict[str, Any], model_cfg: Dict[str, Any]) -> bool:
    task = str(train_cfg.get("task", "")).strip().lower()
    return task == "im2svg" or bool(model_cfg.get("use_vision", False))


def _select_device(device_name: str) -> torch.device:
    def cuda_ok() -> bool:
        if not torch.cuda.is_available():
            return False
        try:
            cap = torch.cuda.get_device_capability(0)
            arch = f"sm_{cap[0]}{cap[1]}"
            arch_list = torch.cuda.get_arch_list()
            if arch_list and arch not in arch_list:
                return False
        except Exception:
            return False
        try:
            x = torch.randn(2, 2, device="cuda")
            _ = (x @ x).cpu()
            torch.cuda.synchronize()
            return True
        except Exception:
            return False

    if device_name == "auto":
        return torch.device("cuda" if cuda_ok() else "cpu")
    if device_name.startswith("cuda") and not cuda_ok():
        return torch.device("cpu")
    return torch.device(device_name)


def _copy_run_artifacts(run_dir: Path, train_cfg: Dict[str, Any], model_cfg: Dict[str, Any], data_cfg: Dict[str, Any], tokenizer) -> None:
    cfg_dir = ensure_dir(run_dir / "configs")
    dump_yaml(train_cfg, cfg_dir / "train.yaml")
    dump_yaml(model_cfg, cfg_dir / "model.yaml")
    dump_yaml(data_cfg, cfg_dir / "data.yaml")
    tokenizer.save(run_dir / "tokenizer.json")


def run_training(
    train_config_path: str | Path,
    model_config_path: str | Path | None = None,
    data_config_path: str | Path | None = None,
    fresh: bool = False,
) -> None:
    train_cfg = load_yaml(train_config_path)
    model_cfg_path, data_cfg_path = _resolve_cfg_paths(train_config_path, model_config_path, data_config_path)
    model_cfg = load_yaml(model_cfg_path)
    data_cfg = load_yaml(data_cfg_path)
    use_im2svg = _use_im2svg(train_cfg, model_cfg)
    if use_im2svg:
        model_cfg["use_vision"] = True

    run_name = str(train_cfg.get("run_name", "yd_vector_run"))
    run_dir = ensure_dir(REPO_ROOT / "outputs" / "runs" / run_name)
    logger = setup_logger("yd_vector.train", log_file=run_dir / "train.log")

    set_seed(int(train_cfg.get("seed", 42)))
    device = _select_device(str(train_cfg.get("device", "auto")))
    logger.info("Run: %s | device=%s", run_name, device)

    manifest_path = _resolve_repo_path(train_cfg.get("manifest_path"))
    if manifest_path is not None and not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    splits_path = _resolve_repo_path(train_cfg.get("splits_path"))
    if splits_path is not None and splits_path.exists():
        logger.info("Split metadata: %s", read_json(splits_path).get("counts", {}))

    im2svg_manifest_path = None
    im2svg_splits_path = None
    if use_im2svg:
        im2svg_manifest_path = _resolve_repo_path(
            train_cfg.get("im2svg_manifest_path", "data_local/manifest/im2svg_manifest_256.jsonl")
        )
        im2svg_splits_path = _resolve_repo_path(
            train_cfg.get("im2svg_splits_path", "data_local/manifest/im2svg_splits.json")
        )
        if im2svg_manifest_path is None or not im2svg_manifest_path.exists():
            raise FileNotFoundError(f"IM2SVG manifest not found: {im2svg_manifest_path}")
        if im2svg_splits_path is None or not im2svg_splits_path.exists():
            raise FileNotFoundError(f"IM2SVG splits not found: {im2svg_splits_path}")
        logger.info("IM2SVG split metadata: %s", read_json(im2svg_splits_path).get("counts", {}))
    elif manifest_path is None:
        raise ValueError("manifest_path is required for SVG-only training")

    tok_cfg = dict(train_cfg.get("tokenizer", {}))
    if tok_cfg.get("vocab_path"):
        vp = Path(tok_cfg["vocab_path"])
        if not vp.is_absolute():
            tok_cfg["vocab_path"] = str(REPO_ROOT / vp)
    tokenizer = build_tokenizer(tok_cfg, manifest_path=str(manifest_path) if manifest_path is not None else None)

    model_cfg["vocab_size"] = tokenizer.vocab_size
    model = YDVectorForCausalLM(ModelConfig.from_dict(model_cfg)).to(device)
    init_checkpoint_path = _resolve_repo_path(train_cfg.get("init_checkpoint_path"))
    init_ignore_prefixes = [str(prefix) for prefix in train_cfg.get("init_ignore_prefixes", [])]

    lr = float(train_cfg.get("lr", 3e-4))
    wd = float(train_cfg.get("weight_decay", 0.01))
    max_steps = int(train_cfg.get("max_steps", 1000))
    warmup_steps = int(train_cfg.get("warmup_steps", 100))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
    batch_size = int(train_cfg.get("batch_size", 4))
    eval_every = int(train_cfg.get("eval_every", 200))
    save_every = int(train_cfg.get("save_every", 200))
    log_every = int(train_cfg.get("log_every", 20))
    max_eval_batches = int(train_cfg.get("max_eval_batches", 20))
    max_seq_len = int(train_cfg.get("max_seq_len", model_cfg.get("max_seq_len", 1024)))

    cache_cfg = dict(train_cfg.get("cache", {}))
    cache_enabled = bool(cache_cfg.get("enabled", False))
    cache_dir = Path(cache_cfg.get("dir", "data_local/cache"))
    if not cache_dir.is_absolute():
        cache_dir = REPO_ROOT / cache_dir

    max_file_bytes = int(data_cfg.get("max_file_bytes", 5 * 1024 * 1024))
    min_chars = int(data_cfg.get("min_chars", 20))
    truncate_chars = int(cache_cfg.get("truncate_chars", data_cfg.get("truncate_chars", 200000)))

    if use_im2svg:
        image_size = int(model_cfg.get("image_size", 256))
        train_dataset = IM2SVGDataset(
            manifest_path=im2svg_manifest_path,
            splits_path=im2svg_splits_path,
            tokenizer=tokenizer,
            split="train",
            image_size=image_size,
            max_seq_len=max_seq_len,
            max_file_bytes=max_file_bytes,
            min_chars=min_chars,
            truncate_chars=truncate_chars,
        )
        val_dataset = IM2SVGDataset(
            manifest_path=im2svg_manifest_path,
            splits_path=im2svg_splits_path,
            tokenizer=tokenizer,
            split="val",
            image_size=image_size,
            max_seq_len=max_seq_len,
            max_file_bytes=max_file_bytes,
            min_chars=min_chars,
            truncate_chars=truncate_chars,
            max_records=max_eval_batches * batch_size * 2,
        )
        collate_fn = im2svg_collate_fn
    else:
        train_dataset = SVGIterableDataset(
            manifest_path=manifest_path,
            tokenizer=tokenizer,
            split="train",
            max_seq_len=max_seq_len,
            max_file_bytes=max_file_bytes,
            min_chars=min_chars,
            truncate_chars=truncate_chars,
            cache_dir=cache_dir,
            cache_enabled=cache_enabled,
            repeat=True,
        )
        val_dataset = SVGIterableDataset(
            manifest_path=manifest_path,
            tokenizer=tokenizer,
            split="val",
            max_seq_len=max_seq_len,
            max_file_bytes=max_file_bytes,
            min_chars=min_chars,
            truncate_chars=truncate_chars,
            cache_dir=cache_dir,
            cache_enabled=cache_enabled,
            repeat=False,
            max_records=max_eval_batches * batch_size * 2,
        )
        collate_fn = lambda b: causal_lm_collate(b, pad_token_id=tokenizer.pad_token_id)

    num_workers = int(train_cfg.get("num_workers", 0))
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    optimizer = build_optimizer(model, lr=lr, weight_decay=wd)
    scheduler = build_scheduler(optimizer, warmup_steps=warmup_steps, max_steps=max_steps)
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = build_scaler(enabled=use_amp)

    _copy_run_artifacts(run_dir, train_cfg, model_cfg, data_cfg, tokenizer)

    global_step = 0
    best_val = None
    resumed = False
    if not fresh:
        state = load_latest_checkpoint(run_dir, model, optimizer=optimizer, scaler=scaler, map_location=device)
        if state is not None:
            global_step = int(state.get("step", 0))
            best_val = state.get("best_val")
            resumed = True
            logger.info("Resumed from checkpoint/latest at step=%d", global_step)
    else:
        latest = run_dir / "checkpoints" / "latest"
        if latest.exists():
            shutil.rmtree(latest)

    if not resumed and init_checkpoint_path is not None:
        init_state = load_checkpoint_weights(
            init_checkpoint_path,
            model,
            map_location=device,
            ignore_prefixes=init_ignore_prefixes,
        )
        logger.info(
            "Initialized model from %s | loaded=%d ignored=%d mismatched=%d missing=%d unexpected=%d",
            init_checkpoint_path,
            init_state["loaded_count"],
            init_state["ignored_count"],
            init_state["shape_mismatch_count"],
            init_state["missing_count"],
            init_state["unexpected_count"],
        )
        if init_state["ignored_keys"]:
            logger.info("Ignored init key sample: %s", init_state["ignored_keys"][:5])
        if init_state["shape_mismatch_keys"]:
            logger.info("Shape-mismatch init key sample: %s", init_state["shape_mismatch_keys"][:5])

    if use_im2svg:
        split_meta = read_json(im2svg_splits_path)
        train_count = int(split_meta.get("counts", {}).get("train", len(train_dataset)))
        val_count = int(split_meta.get("counts", {}).get("val", len(val_dataset)))
    else:
        train_count = count_split_records(manifest_path, split="train")
        val_count = count_split_records(manifest_path, split="val")
    logger.info("Dataset counts | train=%d val=%d", train_count, val_count)
    if train_count == 0:
        raise RuntimeError("No train samples found in manifest.")

    model.train()
    train_iter = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)

    while global_step < max_steps:
        accum_loss = 0.0
        for _ in range(grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            pixel_values = batch.get("pixel_values")
            if pixel_values is not None:
                pixel_values = pixel_values.to(device, non_blocking=True)

            with autocast_context(use_amp):
                loss, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    pixel_values=pixel_values,
                )
                if loss is None:
                    raise RuntimeError("Model did not return loss for training batch.")
                loss = loss / grad_accum_steps

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accum_loss += float(loss.item())

        if use_amp:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        global_step += 1

        if global_step % log_every == 0:
            logger.info(
                "step=%d/%d loss=%.4f ppl=%.2f lr=%.6f",
                global_step,
                max_steps,
                accum_loss,
                loss_to_perplexity(accum_loss),
                optimizer.param_groups[0]["lr"],
            )

        if eval_every > 0 and global_step % eval_every == 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                num_workers=0,
                collate_fn=collate_fn,
                pin_memory=pin_memory,
            )
            metrics = evaluate(model, val_loader, device=device, tokenizer=tokenizer, max_batches=max_eval_batches)
            logger.info(
                "eval step=%d val_loss=%.4f val_ppl=%.2f svg_valid=%.3f",
                global_step,
                metrics["val_loss"],
                metrics["val_perplexity"],
                metrics["val_svg_valid_ratio"],
            )
            if best_val is None or metrics["val_loss"] < float(best_val):
                best_val = metrics["val_loss"]
                logger.info("New best val loss: %.4f", best_val)

        if save_every > 0 and global_step % save_every == 0:
            save_checkpoint(
                run_dir=run_dir,
                step=global_step,
                model=model,
                optimizer=optimizer,
                scaler=scaler if use_amp else None,
                epoch=0,
                best_val=best_val,
            )
            logger.info("Saved checkpoint at step=%d", global_step)

    save_checkpoint(
        run_dir=run_dir,
        step=global_step,
        model=model,
        optimizer=optimizer,
        scaler=scaler if use_amp else None,
        epoch=0,
        best_val=best_val,
    )
    logger.info("Training complete. Final step=%d", global_step)
