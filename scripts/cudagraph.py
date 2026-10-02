"""transformer.forward 全体の CUDA Graph capture/replay (diffusers-ltx2_5 からの移植)。

方式: side stream で warmup → clearCublasWorkspaces → 共有 mempool で capture →
以後は入力コピー + replay。キーは (テンソル引数の shape/dtype/device) +
(非テンソル引数の repr)。未知の型・テンソル入りコンテナは eager に落とす。

Qwen-Image 2.1 での成立条件:
- 全常駐(オフロード無し)・LoRA はマージ済み・CFG=1(負プロンプト経路なし)
- use_kv_cache=False で呼ぶこと。KVキャッシュオブジェクトは capture に
  テンソルアドレスが焼き込まれ、次ジョブで黙って stale になるため扱わない
  (kv_cache が None 以外ならキー化できず自動的に eager へ落ちる)。
"""
from __future__ import annotations

import time
from typing import Any

import torch

_EAGER = object()


class ForwardGraphRunner:
    def __init__(self, module: torch.nn.Module, warmup_iters: int = 3, max_captures: int = 8):
        self._module = module
        self._orig_forward = module.forward
        self._warmup_iters = warmup_iters
        self._max_captures = max_captures
        self._pool = None
        self._captures: dict[tuple, Any] = {}
        self.enabled = True
        self.replays = 0

    def install(self) -> None:
        self._module.forward = self

    def uninstall(self) -> None:
        if self._module.forward is self:
            del self._module.forward  # type: ignore[attr-defined]

    def reset(self) -> None:
        n = sum(1 for v in self._captures.values() if v is not _EAGER)
        self._captures.clear()
        if n:
            torch.cuda.synchronize()
            print(f"cudagraph: reset ({n} captures dropped)", flush=True)

    def _key(self, kwargs: dict) -> tuple | None:
        parts = []
        for k in sorted(kwargs):
            v = kwargs[k]
            if isinstance(v, torch.Tensor):
                parts.append((k, "T", tuple(v.shape), str(v.dtype), str(v.device)))
            elif isinstance(v, (int, float, str, bool, type(None))):
                parts.append((k, "S", v))
            elif isinstance(v, dict):
                if any(isinstance(x, torch.Tensor) for x in v.values()):
                    return None
                parts.append((k, "D", tuple(sorted((dk, repr(dv)) for dk, dv in v.items()))))
            elif isinstance(v, (list, tuple)):
                if any(isinstance(x, torch.Tensor) for x in v):
                    return None
                parts.append((k, "L", repr(v)))
            else:
                return None
        return tuple(parts)

    def __call__(self, *args, **kwargs):
        if not self.enabled or args:
            return self._orig_forward(*args, **kwargs)
        key = self._key(kwargs)
        if key is None:
            return self._orig_forward(**kwargs)
        cap = self._captures.get(key)
        if cap is _EAGER:
            return self._orig_forward(**kwargs)
        if cap is None:
            n_live = sum(1 for v in self._captures.values() if v is not _EAGER)
            if n_live >= self._max_captures:
                self._captures[key] = _EAGER
                return self._orig_forward(**kwargs)
            try:
                cap = self._capture(kwargs)
            except Exception as exc:
                print(f"cudagraph: capture failed; shape stays eager: {exc!r}", flush=True)
                torch.cuda.synchronize()
                self._captures[key] = _EAGER
                return self._orig_forward(**kwargs)
            self._captures[key] = cap
        else:
            for k, static in cap["inputs"].items():
                static.copy_(kwargs[k])
        cap["graph"].replay()
        self.replays += 1
        return cap["outputs"]

    def _capture(self, kwargs: dict) -> dict:
        statics: dict[str, torch.Tensor] = {}
        static_kwargs: dict[str, Any] = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                s = v.clone()
                statics[k] = s
                static_kwargs[k] = s
            else:
                static_kwargs[k] = v
        t0 = time.time()
        with torch.no_grad():
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(self._warmup_iters):
                    self._orig_forward(**static_kwargs)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            torch._C._cuda_clearCublasWorkspaces()
            for k, s in statics.items():
                s.copy_(kwargs[k])
            if self._pool is None:
                self._pool = torch.cuda.graph_pool_handle()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._pool, capture_error_mode="global"):
                outputs = self._orig_forward(**static_kwargs)
        # warmup が膨らませた非graph領域を返す(TEストリーミングとの同居に必須)
        torch.cuda.empty_cache()
        print(f"cudagraph: captured shape in {time.time() - t0:.2f}s "
              f"({len(statics)} input tensors)", flush=True)
        return {"graph": graph, "inputs": statics, "outputs": outputs}
