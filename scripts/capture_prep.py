"""Qwen-Image 2.1 transformer.forward の CUDA Graph 対応版への差し替え。

オリジナル forward は毎回、ホスト同期を伴う前処理を行う:
- pos_embed(rope): image_pad_mask.tolist() / list.index()
- build_token_metadata: nonzero(出力形状がデータ依存)
- prefix_len: int(tensor.sum())
- _qwenimage21_prefix_segments: tolist()
- joint_hidden_states[:, mask] = ... のブールマスク代入(内部で nonzero)

これらは全て (img_shapes, シーケンス長) ごとに不変なので、初回に eager で
計算してキャッシュし、以後はキャプチャ安全な index_select / index_copy と
キャッシュ済みテンソルだけで前処理を再現する。

制約(満たさない呼び出しはオリジナル forward へ委譲する):
- kv_cache なし(use_kv_cache=False で呼ぶこと)
- encoder_hidden_states_mask が None か全1(バッチ1の通常経路)
- flex attention プロセッサ未使用(既定の SDPA プロセッサ)
"""
from __future__ import annotations

import math
from typing import Any

import torch

from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21FlexAttnProcessor,
    _qwenimage21_prefix_segments,
)

_IMG_TOKENS_PER_SLOT = 4


def install_capture_prep(model) -> None:
    """model.forward をキャッシュ前処理版に差し替える。冪等。"""
    if getattr(model, "_capture_prep_installed", False):
        return
    orig_forward = model.forward
    cache: dict[tuple, dict[str, Any]] = {}

    def _build(img_shapes, img_mask, device):
        repeats = torch.where(img_mask, _IMG_TOKENS_PER_SLOT, 1)[0]
        image_pad_mask = torch.repeat_interleave(img_mask[0], repeats)
        src_idx = torch.repeat_interleave(
            torch.arange(img_mask.shape[1], device=device), repeats
        )
        image_positions = image_pad_mask.nonzero(as_tuple=True)[0]
        rotary_emb = model.pos_embed(img_shapes[0], image_pad_mask, device=device)
        image_ids, target_token_mask = model.build_token_metadata(image_pad_mask, img_shapes[0])
        prefix_len = int((~target_token_mask).sum())
        segments = _qwenimage21_prefix_segments(image_ids, prefix_len)
        return {
            "image_pad_mask": image_pad_mask,
            "src_idx": src_idx,
            "image_positions": image_positions,
            "rotary_emb": rotary_emb,
            "target_token_mask": target_token_mask,
            "prefix_len": prefix_len,
            "segments": segments,
        }

    def capture_forward(
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_shapes,
        img_mask: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        attention_kwargs=None,
        kv_cache=None,
        kv_cache_mode=None,
        return_dict: bool = True,
    ):
        mask_trivial = encoder_hidden_states_mask is None or bool(
            encoder_hidden_states_mask.all()
        )
        uses_flex = any(
            isinstance(b.attn.processor, QwenImage21FlexAttnProcessor)
            for b in model.transformer_blocks
        )
        if kv_cache is not None or not mask_trivial or uses_flex:
            return orig_forward(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                img_shapes=img_shapes,
                img_mask=img_mask,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                attention_kwargs=attention_kwargs,
                kv_cache=kv_cache,
                kv_cache_mode=kv_cache_mode,
                return_dict=return_dict,
            )

        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        key = (
            tuple(tuple(s) for s in img_shapes[0]),
            tuple(img_mask.shape),
            hidden_states.shape[1],
            encoder_hidden_states.shape[1],
        )
        prep = cache.get(key)
        if prep is None:
            prep = _build(img_shapes, img_mask, device)
            cache[key] = prep

        hs = model.img_in(hidden_states)
        enc = model.txt_in(encoder_hidden_states)

        target_tokens = math.prod(img_shapes[0][-1])
        base = torch.cat(
            [enc, enc.new_zeros(batch_size, target_tokens // 4, enc.shape[2])], dim=1
        )
        joint = base.index_select(1, prep["src_idx"]).contiguous()
        joint.index_copy_(1, prep["image_positions"], hs)

        timestep = timestep.to(hs.dtype)
        if model.config.causal_condition:
            timestep = torch.cat([timestep, timestep.new_zeros(1)], dim=0)
            modulation_mask = prep["target_token_mask"]
        else:
            modulation_mask = None
        temb = model.time_text_embed(timestep, hs)
        modulation = model.modulation(temb)

        for block in model.transformer_blocks:
            joint = block(
                hidden_states=joint,
                modulation=modulation,
                rotary_emb=prep["rotary_emb"],
                attention_mask=None,
                target_token_mask=modulation_mask,
                layer_cache=None,
                kv_cache_mode=None,
                cache_write_slice=None,
                segments=prep["segments"],
                key_valid=None,
            )

        joint = model.norm_out(joint, temb, modulation_mask)
        output = model.proj_out(joint)

        if not return_dict:
            return (output,)
        from diffusers.models.modeling_outputs import Transformer2DModelOutput

        return Transformer2DModelOutput(sample=output)

    model.forward = capture_forward
    model._capture_prep_installed = True
