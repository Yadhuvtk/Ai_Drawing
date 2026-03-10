from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image

from yd_vector.data.svg_tokenizer import load_tokenizer
from yd_vector.inference.postprocess import ensure_svg_wrapped
from yd_vector.model.config import ModelConfig
from yd_vector.model.yd_vector_arch import YDVectorForCausalLM
from yd_vector.paths import REPO_ROOT
from yd_vector.utils.io import load_yaml


def _load_prompt(args, cfg) -> str:
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            return f.read()
    if args.prompt is not None:
        return args.prompt
    return str(cfg.get("prompt_svg_prefix", "<svg>"))


def _load_image_tensor(path_value: str | Path, image_size: int, device: torch.device) -> torch.Tensor:
    image_path = Path(path_value)
    if not image_path.is_absolute():
        image_path = REPO_ROOT / image_path
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    if img.size != (image_size, image_size):
        img = img.resize((image_size, image_size), Image.BICUBIC)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).unsqueeze(0).to(device=device)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SVG generation inference.")
    parser.add_argument("--config", type=str, default="configs/infer.yaml")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt_file", type=str, default=None)
    parser.add_argument("--image", type=str, default=None, help="Path to a conditioning image for image-to-SVG models.")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding (temperature=0, top_p=0).")
    args = parser.parse_args()

    infer_cfg_path = Path(args.config)
    if not infer_cfg_path.is_absolute():
        infer_cfg_path = REPO_ROOT / infer_cfg_path
    cfg = load_yaml(infer_cfg_path)

    checkpoint_path = Path(cfg["checkpoint_path"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = REPO_ROOT / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint_path not found: {checkpoint_path}")

    vocab_path = Path(cfg["vocab_path"])
    if not vocab_path.is_absolute():
        vocab_path = REPO_ROOT / vocab_path
    tokenizer = load_tokenizer(vocab_path)

    run_dir = checkpoint_path.parents[2] if len(checkpoint_path.parents) >= 3 else REPO_ROOT
    model_cfg_path = run_dir / "configs" / "model.yaml"
    if not model_cfg_path.exists():
        model_cfg_path = REPO_ROOT / "configs" / "model.yaml"
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

    prompt = _load_prompt(args, cfg)
    input_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False, max_length=config.max_seq_len)
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    image_path = args.image if args.image is not None else cfg.get("image_path", "")
    pixel_values = None
    if config.use_vision:
        if not image_path:
            raise ValueError("An image path is required for inference when config.use_vision is true")
        pixel_values = _load_image_tensor(image_path, image_size=config.image_size, device=device)
    elif image_path:
        raise ValueError("An image path was provided, but the loaded model config does not enable vision conditioning")

    max_new_tokens = int(cfg.get("max_new_tokens", 256))
    temperature = float(cfg.get("temperature", 1.0))
    top_p = float(cfg.get("top_p", 0.9))
    if args.greedy:
        temperature = 0.0
        top_p = 0.0

    out_ids = model.generate(
        input_ids=x,
        pixel_values=pixel_values,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=True)
    text = ensure_svg_wrapped(text)

    output_dir = Path(cfg.get("output_dir", "outputs/infer"))
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_svg = output_dir / f"generated_{ts}.svg"
    with open(out_svg, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)

    print(f"Generated SVG saved to: {out_svg}")
    print(f"Output chars: {len(text)}")


if __name__ == "__main__":
    main()
