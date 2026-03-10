from __future__ import annotations

import argparse
from datetime import datetime
from difflib import SequenceMatcher
import importlib.util
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from yd_vector.data.svg_tokenizer import load_tokenizer
from yd_vector.inference.postprocess import ensure_svg_wrapped
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


def _select_images(render_dir: Path, num_images: int) -> list[Path]:
    pngs = sorted(render_dir.glob("*.png"), key=lambda p: p.stat().st_size)
    if len(pngs) < num_images:
        raise RuntimeError(f"Need at least {num_images} PNGs in {render_dir}, found {len(pngs)}")
    if num_images == 1:
        return [pngs[len(pngs) // 2]]

    indexes = []
    for i in range(num_images):
        idx = round(i * (len(pngs) - 1) / (num_images - 1))
        indexes.append(idx)

    selected = []
    seen = set()
    for idx in indexes:
        path = pngs[idx]
        if path not in seen:
            selected.append(path)
            seen.add(path)
    if len(selected) < num_images:
        for path in pngs:
            if path not in seen:
                selected.append(path)
                seen.add(path)
            if len(selected) == num_images:
                break
    return selected


def _normalize_svg(text: str) -> str:
    return " ".join(text.split())


def _preview(text: str, limit: int = 200) -> str:
    clipped = text[:limit]
    return clipped.replace("\n", " ")


def _compare_outputs(texts: list[str]) -> tuple[bool, bool, list[tuple[int, int, float]]]:
    normalized = [_normalize_svg(text) for text in texts]
    identical = len(set(normalized)) == 1
    pairwise = []
    for i, j in combinations(range(len(normalized)), 2):
        ratio = SequenceMatcher(None, normalized[i], normalized[j]).ratio()
        pairwise.append((i, j, ratio))
    nearly_identical = (not identical) and bool(pairwise) and min(r for _, _, r in pairwise) >= 0.98
    return identical, nearly_identical, pairwise


def _suggest_next_step(cfg: dict, texts: list[str], identical: bool, nearly_identical: bool) -> str:
    max_new_tokens = int(cfg.get("max_new_tokens", 256))
    temperature = float(cfg.get("temperature", 1.0))
    top_p = float(cfg.get("top_p", 0.9))
    avg_len = sum(len(text) for text in texts) / max(1, len(texts))
    checkpoint_path = str(cfg.get("checkpoint_path", ""))

    if not identical and not nearly_identical and avg_len >= 200:
        return "Outputs differ enough to suggest image conditioning may be active. Next step: test on a stronger checkpoint before drawing conclusions."
    if max_new_tokens <= 64:
        return "Increase max_new_tokens to 128 or 256 first. The current outputs are short enough that conditioning differences may be truncated away."
    if "smoke" in checkpoint_path.lower() or avg_len < 200:
        return "The current smoke checkpoint is probably too weak for a meaningful conditioning test. Use a better vision-trained checkpoint next."
    if temperature == 0.0 and top_p == 0.0:
        return "After increasing output length, try a slightly less greedy decode such as temperature=0.7 and top_p=0.9."
    return "Use a better vision-trained checkpoint next. The current outputs are too similar to trust the conditioning signal."


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small image-conditioning sanity check.")
    parser.add_argument("--config", type=str, default="configs/infer.yaml")
    parser.add_argument("--render_dir", type=str, default="data_local/renders_256")
    parser.add_argument("--num_images", type=int, default=3)
    parser.add_argument("--images", nargs="*", default=None)
    args = parser.parse_args()

    infer_cfg_path = _resolve_repo_path(args.config)
    cfg, config, model, tokenizer, device = _load_model_and_cfg(infer_cfg_path)
    if not config.use_vision:
        raise ValueError("This sanity check requires a vision-enabled inference config/checkpoint.")

    prompt = infer_script._load_prompt(SimpleNamespace(prompt=None, prompt_file=None), cfg)
    input_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False, max_length=config.max_seq_len)
    x = torch.tensor([input_ids], dtype=torch.long, device=device)

    if args.images:
        image_paths = [_resolve_repo_path(path) for path in args.images]
    else:
        render_dir = _resolve_repo_path(args.render_dir)
        image_paths = _select_images(render_dir, num_images=args.num_images)

    output_root = _resolve_repo_path(cfg.get("output_dir", "outputs/infer"))
    run_dir = output_root / f"sanity_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    max_new_tokens = int(cfg.get("max_new_tokens", 256))
    temperature = float(cfg.get("temperature", 1.0))
    top_p = float(cfg.get("top_p", 0.9))

    texts: list[str] = []
    out_paths: list[Path] = []
    print(f"Using config: {infer_cfg_path}")
    print(f"Using checkpoint: {cfg['checkpoint_path']}")
    print(f"Using output dir: {run_dir}")
    print("")

    for idx, image_path in enumerate(image_paths, start=1):
        pixel_values = infer_script._load_image_tensor(image_path, image_size=config.image_size, device=device)
        out_ids = model.generate(
            input_ids=x,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pixel_values=pixel_values,
        )
        text = tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=True)
        text = ensure_svg_wrapped(text)

        out_path = run_dir / f"{idx:02d}_{image_path.stem}.svg"
        out_path.write_text(text, encoding="utf-8", newline="\n")

        texts.append(text)
        out_paths.append(out_path)

        print(f"[{idx}] image path: {image_path}")
        print(f"[{idx}] output svg path: {out_path}")
        print(f"[{idx}] output character length: {len(text)}")
        print(f"[{idx}] first 200 chars: {_preview(text)}")
        print("")

    identical, nearly_identical, pairwise = _compare_outputs(texts)
    differ_enough = (not identical) and (not nearly_identical)

    print("Comparison:")
    print(f"- identical: {identical}")
    print(f"- nearly identical: {nearly_identical}")
    print(f"- differ enough to suggest conditioning: {differ_enough}")
    for i, j, ratio in pairwise:
        print(f"- similarity image {i + 1} vs {j + 1}: {ratio:.4f}")
    print("")
    print("Interpretation:")
    if identical:
        print("- All normalized outputs are identical. This does not support active image conditioning.")
    elif nearly_identical:
        print("- Outputs are extremely similar. This is weak evidence for conditioning at best.")
    else:
        print("- Outputs differ beyond near-duplicate level. That suggests image conditioning may be active.")
    print(f"- Next suggestion: {_suggest_next_step(cfg, texts, identical, nearly_identical)}")


if __name__ == "__main__":
    main()
