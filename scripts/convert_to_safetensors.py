import os
import torch
from safetensors.torch import save_file

SRC = r"outputs/runs/yd_vector_small/checkpoints/step_00020000/model.pt"
OUT_DIR = r"outputs/exports/YD-Vector-Model"
OUT_PATH = os.path.join(OUT_DIR, "YD-Vector-Model.safetensors")

os.makedirs(OUT_DIR, exist_ok=True)

ckpt = torch.load(SRC, map_location="cpu")
state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

# Make tensors contiguous
state = {k: v.contiguous() for k, v in state.items()}

# ✅ Break shared memory/tied weights (common: token_embedding.weight <-> lm_head.weight)
if "lm_head.weight" in state and "token_embedding.weight" in state:
    # If they share the same storage, clone one so safetensors can serialize safely
    if state["lm_head.weight"].data_ptr() == state["token_embedding.weight"].data_ptr():
        state["lm_head.weight"] = state["lm_head.weight"].clone()

save_file(state, OUT_PATH, metadata={"project": "YD-Vector-Model", "checkpoint": "step_00020000"})
print("Saved:", OUT_PATH)
print("Num tensors:", len(state))