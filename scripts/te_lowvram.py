"""Qwen3-VL テキストエンコーダの低VRAM化(diffusers-ltx2_5 の手法の移植)。

1. apply_te_diet: embed_tokens と lm_head(非共有、各~1.2GiB)を CPU へ移し、
   forward を「内側の Qwen3VLModel を直接呼ぶ」薄いラッパーに差し替えて
   lm_head 計算(全トークン logits の一時テンソル)も省く。
   QwenImage21Pipeline は outputs.hidden_states[-1] しか読まないため無害。
2. apply_te_stream: 言語モデル36層(bf16 計~16GiB)を pinned host memory に
   常駐させ、エンコード中だけ窓付き(既定2層先読み)で GPU へ流す。
   コピー専用 stream + event 同期 + record_stream は ltx2_5 実装のまま。
   bf16 素の重みなので bnb quant_state の移送は不要(コードは残すが無害)。

両者とも出力はビット一致(ltx2_5 で実測済みの方式)。全常駐構成専用で、
enable_model_cpu_offload との併用は不可。
"""
from __future__ import annotations

import torch


def apply_te_diet(text_encoder) -> float:
    """embed_tokens/lm_head を CPU へ移し、lm_head を通らない forward に差し替える。

    Returns: GPU から解放した重みサイズ(GiB)。冪等。
    """
    embed = text_encoder.model.language_model.embed_tokens
    if getattr(embed, "_te_diet_applied", False):
        return 0.0

    freed = embed.weight.numel() * embed.weight.element_size()

    orig_embed_forward = embed.forward

    def bridged_embed_forward(input_ids: torch.Tensor) -> torch.Tensor:
        return orig_embed_forward(input_ids.to("cpu")).to(input_ids.device)

    embed.to("cpu")
    embed.forward = bridged_embed_forward
    embed._te_diet_applied = True

    # 非共有 lm_head も CPU へ(diet forward は呼ばない)
    lm_head = getattr(text_encoder, "lm_head", None)
    if lm_head is not None and lm_head.weight.device.type != "cpu":
        freed += lm_head.weight.numel() * lm_head.weight.element_size()
        lm_head.to("cpu")

    inner = text_encoder.model  # Qwen3VLModel (language_model + visual)

    def diet_forward(input_ids=None, attention_mask=None, **kwargs):
        kwargs.pop("labels", None)
        return inner(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            **kwargs,
        )

    text_encoder.forward = diet_forward
    return freed / 1024**3


def _collect_movable(module: torch.nn.Module):
    items = []
    for _, p in module.named_parameters(recurse=True):
        items.append((p, "data", p.data))
        qs = getattr(p, "quant_state", None)
        if qs is not None:
            for attr in ("absmax", "code", "offset"):
                t = getattr(qs, attr, None)
                if isinstance(t, torch.Tensor):
                    items.append((qs, attr, t))
            nested = getattr(qs, "state2", None)
            if nested is not None:
                for attr in ("absmax", "code"):
                    t = getattr(nested, attr, None)
                    if isinstance(t, torch.Tensor):
                        items.append((nested, attr, t))
    for name, b in module.named_buffers(recurse=True):
        owner = module
        parts = name.split(".")
        for part in parts[:-1]:
            owner = getattr(owner, part)
        items.append((owner, parts[-1], b))
    return items


class WindowedLayerStreamer:
    """固定リングバッファ方式: window+1 スロットのGPUバッファを使い回し、
    エンコード中の新規アロケーションをゼロにする。

    ltx2_5 の原実装は層ごとに .to(device) で確保し record_stream で遅延解放して
    いたが、CUDA Graph の mempool と同居すると expandable_segments が効かず
    アロケータが断片化して全層分(~14GiB)の予約が積み上がる。固定スロットなら
    アロケータを一切経由しないため、Graph 併用でも常駐は slots×層サイズで一定。

    同期規約:
    - copy_stream は slot_free[slot](そのスロットを最後に使った層の計算完了)を
      待ってからコピーする。
    - 計算ストリームは copied[i](層iのコピー完了)を待ってから層iを実行する。
    - 層iの forward 後、slot_free[slot] を計算ストリームに記録する。
    slots = window+1 により「コピー先スロット」と「実行中の層のスロット」は
    常に別物になる。
    """

    def __init__(self, layers, device: torch.device, window: int = 2):
        self.layers = list(layers)
        self.device = device
        self.window = max(1, int(window))
        self.nslots = self.window + 1
        self.copy_stream = torch.cuda.Stream(device=device)
        self.copied = [torch.cuda.Event() for _ in self.layers]
        self.slot_free = [torch.cuda.Event() for _ in range(self.nslots)]
        self.cpu_snap = []
        pinned_gib = 0.0
        for layer in self.layers:
            snap = []
            for owner, attr, t in _collect_movable(layer):
                cpu = t.detach().to("cpu")
                if not cpu.is_pinned():
                    cpu = cpu.pin_memory()
                pinned_gib += cpu.numel() * cpu.element_size() / 1024**3
                snap.append((owner, attr, cpu))
                self._set(owner, attr, cpu)
            self.cpu_snap.append(snap)
        self.pinned_gib = pinned_gib

        # 全層同形を前提にスロットを確保する(異形ならアサート)
        shapes0 = [(c.shape, c.dtype) for _, _, c in self.cpu_snap[0]]
        for i, snap in enumerate(self.cpu_snap[1:], 1):
            assert [(c.shape, c.dtype) for _, _, c in snap] == shapes0, f"layer {i} shape mismatch"
        self.slots = [
            [torch.empty(c.shape, dtype=c.dtype, device=device) for _, _, c in self.cpu_snap[0]]
            for _ in range(self.nslots)
        ]
        self.slots_gib = sum(
            t.numel() * t.element_size() for bufs in self.slots for t in bufs
        ) / 1024**3
        for ev in self.slot_free:
            ev.record()  # 初回は即時使用可

        for i, layer in enumerate(self.layers):
            layer.register_forward_pre_hook(self._pre(i))
            layer.register_forward_hook(self._post(i))

    @staticmethod
    def _set(owner, attr, tensor):
        if isinstance(owner, torch.nn.Parameter):
            owner.data = tensor
        else:
            setattr(owner, attr, tensor)

    def _onload(self, i: int) -> None:
        slot = self.slots[i % self.nslots]
        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_event(self.slot_free[i % self.nslots])
            for (owner, attr, cpu), buf in zip(self.cpu_snap[i], slot):
                buf.copy_(cpu, non_blocking=True)
                self._set(owner, attr, buf)
            self.copied[i].record(self.copy_stream)

    def _pre(self, i: int):
        def hook(_m, _args):
            torch.cuda.current_stream(self.device).wait_event(self.copied[i])
        return hook

    def _post(self, i: int):
        def hook(_m, _args, _out):
            for owner, attr, cpu in self.cpu_snap[i]:
                self._set(owner, attr, cpu)
            self.slot_free[i % self.nslots].record(torch.cuda.current_stream(self.device))
            nxt = i + self.window
            if nxt < len(self.layers):
                self._onload(nxt)
        return hook

    def begin(self) -> None:
        for i in range(min(self.window, len(self.layers))):
            self._onload(i)

    def end(self) -> None:
        torch.cuda.current_stream(self.device).synchronize()
        for i, snap in enumerate(self.cpu_snap):
            for owner, attr, cpu in snap:
                self._set(owner, attr, cpu)


def apply_te_stream(text_encoder, device: torch.device, window: int = 2) -> float:
    """言語モデル層を窓付きストリーミング化する。diet 適用後でも前でも可。冪等。"""
    if getattr(text_encoder, "_te_stream_applied", False):
        return 0.0

    layers = text_encoder.model.language_model.layers
    streamer = WindowedLayerStreamer(layers, device=device, window=window)
    inner_forward = text_encoder.forward

    def streamed_forward(*args, **kwargs):
        streamer.begin()
        try:
            return inner_forward(*args, **kwargs)
        finally:
            streamer.end()

    text_encoder.forward = streamed_forward
    text_encoder._te_stream_applied = True
    text_encoder._te_streamer = streamer
    return streamer.pinned_gib
