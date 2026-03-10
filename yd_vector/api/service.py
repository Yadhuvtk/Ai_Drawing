from __future__ import annotations

import base64
import io
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from PIL import Image

from yd_vector.data.svg_tokenizer import load_tokenizer
from yd_vector.inference.postprocess import ensure_svg_wrapped
from yd_vector.model.config import ModelConfig
from yd_vector.model.yd_vector_arch import YDVectorForCausalLM
from yd_vector.paths import CONFIGS_DIR, REPO_ROOT, ensure_dir
from yd_vector.utils.io import load_yaml


@dataclass
class InferenceBundle:
    infer_cfg_path: Path
    infer_cfg: dict[str, Any]
    checkpoint_path: Path
    vocab_path: Path
    model_cfg_path: Path
    model_cfg: dict[str, Any]
    config: ModelConfig
    tokenizer: Any
    model: YDVectorForCausalLM
    device: torch.device
    checkpoint_mtime: float
    infer_cfg_mtime: float


_MODEL_CACHE: dict[str, InferenceBundle] = {}
_JOB_STORE: dict[str, dict[str, Any]] = {}
_CACHE_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def relative_repo_path(path: str | Path) -> str:
    try:
        return Path(path).resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(Path(path))


def _humanize_name(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").strip().title()


def _select_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        cap = torch.cuda.get_device_capability(0)
        arch = f"sm_{cap[0]}{cap[1]}"
        arch_list = torch.cuda.get_arch_list()
        if arch_list and arch not in arch_list:
            return torch.device("cpu")
    except Exception:
        return torch.device("cpu")
    try:
        x = torch.randn(2, 2, device="cuda")
        _ = (x @ x).cpu()
        torch.cuda.synchronize()
        return torch.device("cuda")
    except Exception:
        return torch.device("cpu")


def current_device_name() -> str:
    return str(_select_device())


def is_busy() -> bool:
    return _INFERENCE_LOCK.locked()


def _resolve_model_cfg_path(checkpoint_path: Path) -> Path:
    run_dir = checkpoint_path.parents[2] if len(checkpoint_path.parents) >= 3 else REPO_ROOT
    model_cfg_path = run_dir / "configs" / "model.yaml"
    if not model_cfg_path.exists():
        model_cfg_path = CONFIGS_DIR / "model.yaml"
    return model_cfg_path


def _prompt_from_cfg(cfg: dict[str, Any], prompt: str | None = None) -> str:
    if prompt is not None and prompt.strip():
        return prompt
    return str(cfg.get("prompt_svg_prefix", "<svg>"))


def _decode_image_bytes(image_data_url: str) -> bytes:
    if not image_data_url:
        raise ValueError("image_data_url is required")
    payload = image_data_url.split(",", 1)[1] if "," in image_data_url else image_data_url
    try:
        return base64.b64decode(payload)
    except Exception as exc:
        raise ValueError("image_data_url is not valid base64") from exc


def _load_image_tensor_from_bytes(image_bytes: bytes, image_size: int, device: torch.device) -> torch.Tensor:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if img.size != (image_size, image_size):
        img = img.resize((image_size, image_size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).unsqueeze(0).to(device=device)


def list_inference_configs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cfg_path in sorted(CONFIGS_DIR.glob("infer*.yaml")):
        cfg = load_yaml(cfg_path)
        checkpoint_value = cfg.get("checkpoint_path", "")
        checkpoint_path = resolve_repo_path(checkpoint_value) if checkpoint_value else None
        checkpoint_exists = bool(checkpoint_path and checkpoint_path.exists())
        checkpoint_mtime = checkpoint_path.stat().st_mtime if checkpoint_exists else 0.0

        model_cfg_path = _resolve_model_cfg_path(checkpoint_path) if checkpoint_path else CONFIGS_DIR / "model.yaml"
        model_cfg = load_yaml(model_cfg_path) if model_cfg_path.exists() else {}
        use_vision = bool(model_cfg.get("use_vision", False))
        run_name = checkpoint_path.parents[2].name if checkpoint_path and len(checkpoint_path.parents) >= 3 else cfg_path.stem

        rows.append(
            {
                "id": relative_repo_path(cfg_path),
                "name": cfg_path.stem,
                "label": _humanize_name(run_name),
                "run_name": run_name,
                "checkpoint_path": relative_repo_path(checkpoint_path) if checkpoint_path else "",
                "checkpoint_exists": checkpoint_exists,
                "checkpoint_modified": datetime.fromtimestamp(checkpoint_mtime, tz=timezone.utc).isoformat()
                if checkpoint_exists
                else None,
                "model_config_path": relative_repo_path(model_cfg_path),
                "image_conditioned": use_vision,
                "max_new_tokens": int(cfg.get("max_new_tokens", 256)),
                "temperature": float(cfg.get("temperature", 1.0)),
                "top_p": float(cfg.get("top_p", 0.9)),
                "prompt_svg_prefix": str(cfg.get("prompt_svg_prefix", "<svg>")),
                "recommended": checkpoint_exists and use_vision,
                "is_default": cfg_path.name == "infer.yaml",
            }
        )

    rows.sort(
        key=lambda row: (
            0 if row["recommended"] else 1,
            0 if row["checkpoint_exists"] else 1,
            -datetime.fromisoformat(row["checkpoint_modified"]).timestamp() if row["checkpoint_modified"] else 0.0,
            row["name"],
        )
    )
    return rows


def _load_bundle(config_path: str | Path) -> InferenceBundle:
    infer_cfg_path = resolve_repo_path(config_path)
    if not infer_cfg_path.exists():
        raise FileNotFoundError(f"Inference config not found: {infer_cfg_path}")

    infer_cfg_mtime = infer_cfg_path.stat().st_mtime
    infer_cfg = load_yaml(infer_cfg_path)

    checkpoint_path = resolve_repo_path(infer_cfg["checkpoint_path"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint_path not found: {checkpoint_path}")
    checkpoint_mtime = checkpoint_path.stat().st_mtime

    cache_key = str(infer_cfg_path.resolve())
    with _CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None and cached.infer_cfg_mtime == infer_cfg_mtime and cached.checkpoint_mtime == checkpoint_mtime:
            return cached

    vocab_path = resolve_repo_path(infer_cfg["vocab_path"])
    if not vocab_path.exists():
        raise FileNotFoundError(f"vocab_path not found: {vocab_path}")

    tokenizer = load_tokenizer(vocab_path)
    model_cfg_path = _resolve_model_cfg_path(checkpoint_path)
    model_cfg = load_yaml(model_cfg_path)
    model_cfg["vocab_size"] = tokenizer.vocab_size
    config = ModelConfig.from_dict(model_cfg)

    device = _select_device()
    model = YDVectorForCausalLM(config).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()

    bundle = InferenceBundle(
        infer_cfg_path=infer_cfg_path,
        infer_cfg=infer_cfg,
        checkpoint_path=checkpoint_path,
        vocab_path=vocab_path,
        model_cfg_path=model_cfg_path,
        model_cfg=model_cfg,
        config=config,
        tokenizer=tokenizer,
        model=model,
        device=device,
        checkpoint_mtime=checkpoint_mtime,
        infer_cfg_mtime=infer_cfg_mtime,
    )
    with _CACHE_LOCK:
        _MODEL_CACHE[cache_key] = bundle
    return bundle


def run_image_inference(
    config_path: str,
    image_data_url: str,
    image_name: str | None = None,
    prompt: str | None = None,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
) -> dict[str, Any]:
    job_id = uuid.uuid4().hex[:12]
    started_at = _utc_now()
    _JOB_STORE[job_id] = {"job_id": job_id, "status": "running", "started_at": started_at}

    with _INFERENCE_LOCK:
        started = perf_counter()
        try:
            bundle = _load_bundle(config_path)
            if not bundle.config.use_vision:
                raise ValueError("The selected inference config does not enable image conditioning")
            image_bytes = _decode_image_bytes(image_data_url)

            decode_max_new_tokens = int(max_new_tokens or bundle.infer_cfg.get("max_new_tokens", 256))
            decode_temperature = float(temperature if temperature is not None else bundle.infer_cfg.get("temperature", 1.0))
            decode_top_p = float(top_p if top_p is not None else bundle.infer_cfg.get("top_p", 0.9))

            prompt_text = _prompt_from_cfg(bundle.infer_cfg, prompt=prompt)
            input_ids = bundle.tokenizer.encode(
                prompt_text,
                add_bos=True,
                add_eos=False,
                max_length=bundle.config.max_seq_len,
            )
            x = torch.tensor([input_ids], dtype=torch.long, device=bundle.device)

            pixel_values = None
            if bundle.config.use_vision:
                pixel_values = _load_image_tensor_from_bytes(
                    image_bytes,
                    image_size=bundle.config.image_size,
                    device=bundle.device,
                )

            out_ids = bundle.model.generate(
                input_ids=x,
                pixel_values=pixel_values,
                max_new_tokens=decode_max_new_tokens,
                temperature=decode_temperature,
                top_p=decode_top_p,
                eos_token_id=bundle.tokenizer.eos_token_id,
            )
            svg_text = bundle.tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=True)
            svg_text = ensure_svg_wrapped(svg_text)

            output_dir = resolve_repo_path(bundle.infer_cfg.get("output_dir", "outputs/infer"))
            ensure_dir(output_dir)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_svg_path = output_dir / f"api_{ts}_{job_id}.svg"
            with open(output_svg_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(svg_text)

            duration_ms = round((perf_counter() - started) * 1000.0, 2)
            result = {
                "job_id": job_id,
                "status": "completed",
                "started_at": started_at,
                "finished_at": _utc_now(),
                "duration_ms": duration_ms,
                "config_path": relative_repo_path(bundle.infer_cfg_path),
                "config_label": _humanize_name(bundle.checkpoint_path.parents[2].name)
                if len(bundle.checkpoint_path.parents) >= 3
                else bundle.infer_cfg_path.stem,
                "checkpoint_path": relative_repo_path(bundle.checkpoint_path),
                "model_config_path": relative_repo_path(bundle.model_cfg_path),
                "image_name": image_name or "uploaded-image",
                "image_conditioned": bool(bundle.config.use_vision),
                "output_svg_path": relative_repo_path(output_svg_path),
                "output_dir": relative_repo_path(output_dir),
                "svg_char_length": len(svg_text),
                "svg_content": svg_text,
                "svg_url": f"/api/jobs/{job_id}/svg",
                "prompt": prompt_text,
                "decode": {
                    "max_new_tokens": decode_max_new_tokens,
                    "temperature": decode_temperature,
                    "top_p": decode_top_p,
                },
                "device": str(bundle.device),
            }
            _JOB_STORE[job_id] = result
            return result
        except Exception as exc:
            failed = {
                "job_id": job_id,
                "status": "failed",
                "started_at": started_at,
                "finished_at": _utc_now(),
                "error": str(exc),
            }
            _JOB_STORE[job_id] = failed
            raise


def get_job(job_id: str) -> dict[str, Any] | None:
    return _JOB_STORE.get(job_id)
