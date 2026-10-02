"""Viggle Turbo LoRA (r128) を Nunchaku 量子化済み Qwen-Image 2.1 へマージする。

- 量子化7層(to_q/k/v/to_out.0/img_mlp.{proj,gate_layer,out}):
  SVDQuant の低ランク枝 (proj_down/proj_up, r32) に LoRA A/B を連結して r160 にする。
  チェックポイントの smooth_factor は全層 1(実測)なので A の補正は不要。
  packed レイアウトは nunchaku の pack/unpack ユーティリティで往復する。
- 非量子化層 (modulation.1, time_text_embed.timestep_embedder.linear_1/2):
  W += B @ A を直接フュージョンする(diffusers同様、alpha情報なし = scale 1.0)。
"""
from __future__ import annotations

from pathlib import Path

import safetensors.torch as st
import torch

from nunchaku.lora.flux.nunchaku_converter import pack_lowrank_weight, unpack_lowrank_weight
from nunchaku.models.linear import SVDQW4A4Linear

PREFIX = "transformer."


def merge_viggle_lora(transformer: torch.nn.Module, lora_path: str | Path, scale: float = 1.0) -> dict:
    lora = st.load_file(str(lora_path))
    module_names = sorted({k.removeprefix(PREFIX).rsplit(".lora_", 1)[0] for k in lora})

    stats = {"svdq": 0, "fused": 0, "skipped": []}
    for name in module_names:
        a = lora[f"{PREFIX}{name}.lora_A.weight"]  # [r, in]
        b = lora[f"{PREFIX}{name}.lora_B.weight"]  # [out, r]
        module = transformer.get_submodule(name)
        dev = next(p.device for p in module.parameters())
        a = a.to(dev, torch.bfloat16) * scale
        b = b.to(dev, torch.bfloat16)

        if isinstance(module, SVDQW4A4Linear):
            pd = unpack_lowrank_weight(module.proj_down.data, down=True)   # [r0, in]
            pu = unpack_lowrank_weight(module.proj_up.data, down=False)   # [out, r0]
            pd_new = torch.cat([pd, a], dim=0).contiguous()
            pu_new = torch.cat([pu, b], dim=1).contiguous()
            new_rank = pd_new.shape[0]
            assert new_rank % 16 == 0, f"rank {new_rank} not multiple of 16 for {name}"
            module.proj_down.data = pack_lowrank_weight(pd_new, down=True)
            module.proj_up.data = pack_lowrank_weight(pu_new, down=False)
            module.rank = new_rank
            stats["svdq"] += 1
        elif isinstance(module, torch.nn.Linear):
            module.weight.data += (b.float() @ a.float()).to(module.weight.dtype)
            stats["fused"] += 1
        else:
            stats["skipped"].append(name)
    return stats
