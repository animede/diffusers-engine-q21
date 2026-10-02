"""Nunchaku SVDQuant NVFP4 wrapper for Qwen-Image 2.1.

公式nunchaku(1.3.0.dev20260306)には2.1用クラスが無いため、
diffusersの QwenImage21Transformer2DModel のLinearを SVDQW4A4Linear に
差し替える最小ラッパーを提供する。forwardはdiffusers実装をそのまま使う
(2.1はブロック毎のQKV融合なし・ブロック内AdaNorm無しなので差し替えだけで済む)。

チェックポイント形式: catplusplus/nunchaku-qwen-image-2.1 (svdq-fp4_r32,
per-channel wcscales)。各量子化層は qweight/wscales/wcscales/bias/
smooth_factor/smooth_factor_orig/proj_down/proj_up を持つ。
"""
from __future__ import annotations

import json
from pathlib import Path

import safetensors.torch as st
import torch

from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21Rope,
    QwenImage21TemporalTimesteps,
    QwenImage21Transformer2DModel,
)
from nunchaku.models.linear import SVDQW4A4Linear
from nunchaku.models.transformers.utils import patch_scale_key

QUANTIZED_LINEARS = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "img_mlp.proj",
    "img_mlp.gate_layer",
    "img_mlp.out",
)


def _replace_linear(parent: torch.nn.Module, name: str, rank: int, torch_dtype: torch.dtype) -> None:
    linear = getattr(parent, name) if not name.isdigit() else parent[int(name)]
    quantized = SVDQW4A4Linear(
        in_features=linear.in_features,
        out_features=linear.out_features,
        rank=rank,
        bias=True,  # チェックポイントはbias無し層にもゼロbiasを持つ
        precision="nvfp4",
        torch_dtype=torch_dtype,
        device=linear.weight.device,
    )
    if name.isdigit():
        parent[int(name)] = quantized
    else:
        setattr(parent, name, quantized)


def load_nunchaku_qwen21_transformer(
    checkpoint_path: str | Path,
    config_path: str | Path,
    device: str | torch.device = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> QwenImage21Transformer2DModel:
    """量子化済みチェックポイントから2.1 DiTを構築してGPU常駐で返す。"""
    checkpoint_path = Path(checkpoint_path)
    config = json.loads(Path(config_path).read_text())
    config.pop("quantization_config", None)
    rank = 32
    if checkpoint_path.with_name("config.json").exists():
        meta = json.loads(checkpoint_path.with_name("config.json").read_text())
        rank = meta.get("quantization_config", {}).get("rank", 32)

    with torch.device("meta"):
        model = QwenImage21Transformer2DModel.from_config(config)
    model = model.to(torch_dtype)

    for block in model.transformer_blocks:
        for path in QUANTIZED_LINEARS:
            *parents, leaf = path.split(".")
            parent = block
            for p in parents:
                parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
            _replace_linear(parent, leaf, rank=rank, torch_dtype=torch_dtype)

    model = model.to_empty(device=device)

    # metaデバイス初期化で失われた非永続バッファ/定数を再計算する
    model.pos_embed = QwenImage21Rope(theta=10000, axes_dim=list(config.get("axes_dims_rope", [16, 56, 56])))
    for module in model.modules():
        if isinstance(module, QwenImage21TemporalTimesteps):
            half = module.timestep_dim // 2
            import math

            freqs = torch.exp(
                -math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32) / half
            )
            module.freqs = freqs.to(device)

    state_dict = st.load_file(str(checkpoint_path))
    patch_scale_key(model, state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in missing if "freqs" not in k]
    if missing or unexpected:
        raise RuntimeError(f"state dict mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    return model
