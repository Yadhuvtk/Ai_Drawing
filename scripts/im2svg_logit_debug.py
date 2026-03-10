from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yd_vector.data.svg_tokenizer import load_tokenizer
from yd_vector.model.config import ModelConfig
from yd_vector.model.yd_vector_arch import YDVectorForCausalLM
from yd_vector.paths import REPO_ROOT
from yd_vector.utils.io import load_yaml

_INFER_SPEC = importlib.util.spec_from_file_location("yd_vector_scripts_infer", Path(__file__).with_name("infer.py"))
if _INFER_SPEC is None or _INFER_SPEC.loader is None:
    raise ImportError(f"Could not load infer.py from {Path(__file__).with_name('infer.py')}")
infer_script = importlib.util.module_from_spec(_INFER_SPEC)
_INFER_SPEC.loader.exec_module(infer_script)


def _resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _load_model_and_cfg(config_path: Path) -> tuple[dict, ModelConfig, YDVectorForCausalLM, object, torch.device]:
    cfg = load_yaml(config_path)

    checkpoint_path = _resolve_repo_path(cfg["checkpoint_path"])
    vocab_path = _resolve_repo_path(cfg["vocab_path"])
    tokenizer = load_tokenizer(vocab_path)

    run_dir = checkpoint_path.parents[2] if len(checkpoint_path.parents) >= 3 else REPO_ROOT
    model_cfg_path = run_dir / "configs" / "model.yaml"
    if not model_cfg_path.exists():
        model_cfg_path = REPO_ROOT / "configs" / "model.yaml"
    model_cfg = load_yaml(model_cfg_path)
    model_cfg["vocab_size"] = tokenizer.vocab_size
    config = ModelConfig.from_dict(model_cfg)

    device = infer_script._select_device()
    model = YDVectorForCausalLM(config).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    return cfg, config, model, tokenizer, device


def _top5(logits: torch.Tensor, tokenizer) -> list[tuple[int, float, str]]:
    values, indices = torch.topk(logits, k=5)
    rows = []
    for idx, value in zip(indices.tolist(), values.tolist()):
        token = tokenizer.decode([idx], skip_special_tokens=False)
        rows.append((idx, float(value), repr(token)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare next-token logits for two different conditioning images.")
    parser.add_argument("--config", type=str, default="configs/infer.yaml")
    parser.add_argument("--image_a", type=str, required=True)
    parser.add_argument("--image_b", type=str, required=True)
    args = parser.parse_args()

    infer_cfg_path = _resolve_repo_path(args.config)
    cfg, config, model, tokenizer, device = _load_model_and_cfg(infer_cfg_path)
    if not config.use_vision:
        raise ValueError("This diagnostic requires a vision-enabled inference config/checkpoint.")

    prompt = infer_script._load_prompt(SimpleNamespace(prompt=None, prompt_file=None), cfg)
    input_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False, max_length=config.max_seq_len)
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(x, dtype=torch.long, device=device)

    image_a_path = _resolve_repo_path(args.image_a)
    image_b_path = _resolve_repo_path(args.image_b)
    pixel_values_a = infer_script._load_image_tensor(image_a_path, image_size=config.image_size, device=device)
    pixel_values_b = infer_script._load_image_tensor(image_b_path, image_size=config.image_size, device=device)

    with torch.no_grad():
        _, logits_a = model(input_ids=x, attention_mask=attention_mask, labels=None, pixel_values=pixel_values_a)
        _, logits_b = model(input_ids=x, attention_mask=attention_mask, labels=None, pixel_values=pixel_values_b)

    next_logits_a = logits_a[0, -1, :].detach().float().cpu()
    next_logits_b = logits_b[0, -1, :].detach().float().cpu()
    diff = (next_logits_a - next_logits_b).abs()

    exactly_equal = bool(torch.equal(next_logits_a, next_logits_b))
    max_abs_diff = float(diff.max().item())
    mean_abs_diff = float(diff.mean().item())

    print(f"config: {infer_cfg_path}")
    print(f"image_a: {image_a_path}")
    print(f"image_b: {image_b_path}")
    print("")
    print(f"logits exactly equal: {exactly_equal}")
    print(f"max absolute difference: {max_abs_diff:.10f}")
    print(f"mean absolute difference: {mean_abs_diff:.10f}")
    print("")
    print("top-5 next-token predictions for image_a:")
    for idx, value, token in _top5(next_logits_a, tokenizer):
        print(f"- id={idx} logit={value:.6f} token={token}")
    print("")
    print("top-5 next-token predictions for image_b:")
    for idx, value, token in _top5(next_logits_b, tokenizer):
        print(f"- id={idx} logit={value:.6f} token={token}")
    print("")
    print("interpretation:")
    if exactly_equal:
        print("- The image signal is effectively ignored at the compared next-token position.")
    elif max_abs_diff < 1e-5 and mean_abs_diff < 1e-6:
        print("- The image signal is present but extremely weak at the compared next-token position.")
    elif max_abs_diff < 1e-3 and mean_abs_diff < 1e-4:
        print("- The image signal is weak but measurable at the compared next-token position.")
    else:
        print("- The image signal is strong enough to change next-token logits in a noticeable way.")


if __name__ == "__main__":
    main()
