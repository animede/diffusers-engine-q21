from __future__ import annotations

import gc
import json
import secrets
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import Any

from PIL import Image

from .settings import (
    ACCELERATION,
    COMPILE_BLOCKS,
    CPU_OFFLOAD,
    CUDA_DEVICE,
    MODEL_DIR,
    NVFP4_CACHE_DIR,
    OUTPUT_DIR,
    QUANTIZATION,
    REFERENCE_MODE,
    START_PROFILE,
    TE_DIET,
    TE_STREAM,
    TE_STREAM_WINDOW,
    TRIM_CUDA_CACHE,
    VAE_TILING,
    VIGGLE_LORA_DIR,
)


VIGGLE_WEIGHT_NAME = "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors"
VIGGLE_SIGMAS = [1.0, 0.9375, 0.875, 0.75, 0.5, 0.25]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def reference_resolution(image_count: int) -> int:
    """Keep multi-reference vision encoding inside a 24 GiB VRAM budget."""
    if REFERENCE_MODE == "full" or image_count <= 2:
        return 1024
    # Hold the total vision-token budget near 2.5 full-resolution images.
    scaled = int(1024 * (2.5 / image_count) ** 0.5)
    return max(512, min(1024, scaled // 64 * 64))


@dataclass
class Job:
    id: str
    status: str = "queued"
    progress: float = 0.0
    step: int = 0
    total_steps: int = 0
    message: str = "待機中"
    created_at: str = field(default_factory=utc_now)
    started_at: str | None = None
    finished_at: str | None = None
    output_url: str | None = None
    metadata_url: str | None = None
    error: str | None = None


class GenerationService:
    """Owns one pipeline and serializes GPU jobs through one worker thread."""

    def __init__(self) -> None:
        self.pipeline: Any | None = None
        self.model_status = "not_loaded"
        self.model_error: str | None = None
        self.model_lock = threading.Lock()
        self.jobs_lock = threading.Lock()
        self.jobs: dict[str, Job] = {}
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen-worker")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def model_downloaded(self) -> bool:
        return (MODEL_DIR / "model_index.json").is_file() and (MODEL_DIR / ".download-complete").is_file()

    def model_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "status": self.model_status,
            "downloaded": self.model_downloaded,
            "path": str(MODEL_DIR),
            "device": CUDA_DEVICE,
            "cpu_offload": CPU_OFFLOAD,
            "quantization": QUANTIZATION,
            "acceleration": ACCELERATION,
            "effective_steps": 6 if ACCELERATION == "viggle-r128" else None,
            "max_reference_images": 3 if ACCELERATION == "viggle-r128" else 10,
            "start_profile": START_PROFILE,
            "reference_mode": REFERENCE_MODE,
            "vae_tiling": VAE_TILING,
            "trim_cuda_cache": TRIM_CUDA_CACHE,
            "nvfp4_cache": (NVFP4_CACHE_DIR / ".complete").is_file(),
            "error": self.model_error,
        }
        try:
            import torch

            info["torch_version"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["gpus"] = [
                    {
                        "index": index,
                        "name": torch.cuda.get_device_name(index),
                        "memory_gb": round(
                            torch.cuda.get_device_properties(index).total_memory / 1024**3, 1
                        ),
                    }
                    for index in range(torch.cuda.device_count())
                ]
        except Exception as exc:  # Health should still be available when torch is broken.
            info["cuda_available"] = False
            info["runtime_error"] = str(exc)
        return info

    def request_load(self) -> str:
        if self.pipeline is not None:
            return "ready"
        self.executor.submit(self.ensure_model)
        return "queued"

    def ensure_model(self) -> Any:
        if self.pipeline is not None:
            return self.pipeline

        with self.model_lock:
            if self.pipeline is not None:
                return self.pipeline
            if not self.model_downloaded:
                raise RuntimeError(
                    f"モデルが見つかりません: {MODEL_DIR}。先に scripts/download_model.py を実行してください。"
                )

            self.model_status = "loading"
            self.model_error = None
            try:
                import torch
                from diffusers import QwenImage21Pipeline

                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA対応GPUが見つかりません。")

                torch.cuda.set_device(CUDA_DEVICE)

                load_args: dict[str, Any] = {
                    "dtype": torch.bfloat16,
                    "local_files_only": True,
                }
                if QUANTIZATION in {"nvfp4", "fp8"}:
                    from diffusers import PipelineQuantizationConfig
                    from diffusers import TorchAoConfig as DiffusersTorchAoConfig
                    from transformers import TorchAoConfig as TransformersTorchAoConfig

                    if QUANTIZATION == "nvfp4":
                        if torch.cuda.get_device_capability() < (10, 0):
                            raise RuntimeError(
                                "NVFP4はBlackwell GPU専用です。QWEN_QUANTIZATION=fp8 または bf16を指定してください。"
                            )
                        if CPU_OFFLOAD:
                            raise RuntimeError(
                                "NVFP4とCPUオフロードは併用できません。24GB BlackwellではQWEN_CPU_OFFLOAD=0のまま使用できます。"
                            )
                        try:
                            import mslk  # noqa: F401
                            from torchao.prototype.mx_formats import (
                                NVFP4DynamicActivationNVFP4WeightConfig,
                            )
                        except ImportError as exc:
                            raise RuntimeError(
                                "NVFP4に必要なTorchAO/MSLKがありません。requirements.txtを再インストールしてください。"
                            ) from exc

                        from diffusers import QwenImage21Transformer2DModel
                        from transformers import Qwen3VLForConditionalGeneration

                        cache_ready = (NVFP4_CACHE_DIR / ".complete").is_file()
                        # img_in can receive non-contiguous tensors and Qwen3-VL's vision MLP has a
                        # post-GLU width not divisible by 16. Keep only those incompatible paths in BF16.
                        transformer_source = MODEL_DIR / "transformer"
                        text_encoder_source = MODEL_DIR / "text_encoder"
                        transformer_load_args: dict[str, Any] = {}
                        text_encoder_load_args: dict[str, Any] = {}
                        if cache_ready:
                            transformer_source = NVFP4_CACHE_DIR / "transformer"
                            text_encoder_source = NVFP4_CACHE_DIR / "text_encoder"
                        else:
                            transformer_load_args["quantization_config"] = DiffusersTorchAoConfig(
                                NVFP4DynamicActivationNVFP4WeightConfig(),
                                modules_to_not_convert=["img_in"],
                            )
                            text_encoder_load_args["quantization_config"] = TransformersTorchAoConfig(
                                NVFP4DynamicActivationNVFP4WeightConfig(),
                                modules_to_not_convert=["model.visual"],
                            )

                        # Materialize and trim one component at a time. Besides honoring cuda:N,
                        # this prevents conversion workspaces accumulating during an uncached load.
                        load_args["transformer"] = QwenImage21Transformer2DModel.from_pretrained(
                            transformer_source,
                            dtype=torch.bfloat16,
                            local_files_only=True,
                            device_map=CUDA_DEVICE,
                            **transformer_load_args,
                        )
                        gc.collect()
                        torch.cuda.empty_cache()
                        load_args["text_encoder"] = Qwen3VLForConditionalGeneration.from_pretrained(
                            text_encoder_source,
                            dtype=torch.bfloat16,
                            local_files_only=True,
                            device_map=CUDA_DEVICE,
                            **text_encoder_load_args,
                        )
                        gc.collect()
                        torch.cuda.empty_cache()
                    else:
                        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig

                        transformer_config = DiffusersTorchAoConfig(
                            Float8DynamicActivationFloat8WeightConfig()
                        )
                        text_encoder_config = TransformersTorchAoConfig(
                            Float8DynamicActivationFloat8WeightConfig()
                        )

                        # Diffusers owns the DiT while Transformers owns Qwen3-VL.
                        load_args["quantization_config"] = PipelineQuantizationConfig(
                            quant_mapping={
                                "transformer": transformer_config,
                                "text_encoder": text_encoder_config,
                            }
                        )

                pipe = QwenImage21Pipeline.from_pretrained(MODEL_DIR, **load_args)
                if CPU_OFFLOAD:
                    gpu_id = int(CUDA_DEVICE.rsplit(":", 1)[1]) if ":" in CUDA_DEVICE else 0
                    pipe.enable_model_cpu_offload(gpu_id=gpu_id)
                elif QUANTIZATION == "nvfp4":
                    pipe.vae.to(CUDA_DEVICE)
                    gc.collect()
                    torch.cuda.empty_cache()
                else:
                    pipe.to(CUDA_DEVICE)
                if ACCELERATION == "viggle-r128":
                    from diffusers import FlowMatchEulerDiscreteScheduler

                    weight_path = VIGGLE_LORA_DIR / VIGGLE_WEIGHT_NAME
                    scheduler_path = VIGGLE_LORA_DIR / "scheduler" / "scheduler_config.json"
                    if not weight_path.is_file() or not scheduler_path.is_file():
                        raise RuntimeError(
                            f"Viggle Turbo r128が見つかりません: {VIGGLE_LORA_DIR}。"
                            "scripts/download_viggle.py を実行してください。"
                        )
                    pipe.load_lora_weights(
                        VIGGLE_LORA_DIR,
                        weight_name=VIGGLE_WEIGHT_NAME,
                        adapter_name="viggle-turbo",
                    )
                    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                        VIGGLE_LORA_DIR,
                        subfolder="scheduler",
                        local_files_only=True,
                    )
                if TE_DIET or TE_STREAM:
                    # tediet: TEのembed/lm_headをCPUへ、層をpinnedホストから
                    # ストリーミング。出力はビット一致(te-diet README参照)。
                    from tediet import apply_diet, apply_stream

                    if TE_DIET:
                        freed = apply_diet(
                            pipe.text_encoder,
                            embed_path="model.language_model.embed_tokens",
                        )
                        print(f"[q21] tediet: diet freed {freed:.2f} GiB", flush=True)
                    if TE_STREAM:
                        pinned = apply_stream(
                            pipe.text_encoder,
                            "model.language_model.layers",
                            device=CUDA_DEVICE,
                            window=TE_STREAM_WINDOW,
                        )
                        print(
                            f"[q21] tediet: streaming {pinned:.2f} GiB from pinned host "
                            f"(window={TE_STREAM_WINDOW})",
                            flush=True,
                        )
                    gc.collect()
                    torch.cuda.empty_cache()
                if COMPILE_BLOCKS:
                    # dynamic=False の静的カーネルが最速(実測3.9秒/1024²)だが、
                    # プロンプト長が変わるたびに約7秒の再コンパイルが走る。dynamic
                    # カーネルは eager より遅い(実測8.5秒)ため使わない。代わりに
                    # プロンプト埋め込みを64トークン単位へパディングしてシェイプを
                    # バケット化し、再コンパイルを「バケットごとに1回」に抑える。
                    # パディング位置は prompt_embeds_mask が attention から除外する
                    # (バッチ生成時に diffusers 自身が行うパディングと同じ経路)。
                    torch._dynamo.config.cache_size_limit = 256
                    pipe.transformer.compile_repeated_blocks(mode="default", dynamic=False)

                    bucket = 64
                    original_encode_for_pad = pipe.encode_prompt

                    @wraps(original_encode_for_pad)
                    def encode_prompt_padded(*args: Any, **kwargs: Any):
                        prompt_embeds, prompt_embeds_mask, image_pad_mask = (
                            original_encode_for_pad(*args, **kwargs)
                        )
                        seq_len = prompt_embeds.shape[1]
                        padded = ((seq_len + bucket - 1) // bucket) * bucket
                        if padded != seq_len:
                            pad = padded - seq_len
                            prompt_embeds = torch.nn.functional.pad(
                                prompt_embeds, (0, 0, 0, pad)
                            )
                            if prompt_embeds_mask is None:
                                prompt_embeds_mask = prompt_embeds.new_ones(
                                    prompt_embeds.shape[:2], dtype=torch.bool
                                )
                                prompt_embeds_mask[:, seq_len:] = False
                            else:
                                prompt_embeds_mask = torch.nn.functional.pad(
                                    prompt_embeds_mask, (0, pad)
                                )
                            image_pad_mask = torch.nn.functional.pad(
                                image_pad_mask, (0, pad)
                            )
                        return prompt_embeds, prompt_embeds_mask, image_pad_mask

                    pipe.encode_prompt = encode_prompt_padded
                    print(
                        "[q21] regional compile enabled "
                        f"(static kernels; prompts padded to {bucket}-token buckets, "
                        "first job per bucket pays the compile)",
                        flush=True,
                    )
                if VAE_TILING and hasattr(pipe.vae, "enable_tiling"):
                    pipe.vae.enable_tiling()
                if TRIM_CUDA_CACHE:
                    original_encode_prompt = pipe.encode_prompt

                    @wraps(original_encode_prompt)
                    def encode_prompt_and_trim_cache(*args: Any, **kwargs: Any):
                        result = original_encode_prompt(*args, **kwargs)
                        # Return temporary vision memory before denoising and VAE decoding.
                        torch.cuda.empty_cache()
                        return result

                    pipe.encode_prompt = encode_prompt_and_trim_cache

                self.pipeline = pipe
                self.model_status = "ready"
                return pipe
            except Exception as exc:
                self.model_status = "error"
                self.model_error = str(exc)
                raise

    def unload(self) -> None:
        with self.jobs_lock:
            busy = any(job.status in {"queued", "loading_model", "running"} for job in self.jobs.values())
        if busy:
            raise RuntimeError("生成ジョブの実行中はモデルを解放できません。")
        with self.model_lock:
            self.pipeline = None
            self.model_status = "not_loaded"
            self.model_error = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def submit(self, params: dict[str, Any], images: list[Image.Image]) -> Job:
        params = dict(params)
        if ACCELERATION == "viggle-r128":
            params["requested_steps"] = params["steps"]
            params["requested_guidance_scale"] = params["guidance_scale"]
            params["steps"] = 6
            params["guidance_scale"] = 1.0
        job = Job(id=uuid.uuid4().hex, total_steps=params["steps"])
        with self.jobs_lock:
            self.jobs[job.id] = job
            self._prune_jobs_locked()
        self.executor.submit(self._run_job, job.id, params, images)
        return job

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.jobs_lock:
            job = self.jobs.get(job_id)
            return asdict(job) if job else None

    def _update_job(self, job_id: str, **updates: Any) -> None:
        with self.jobs_lock:
            job = self.jobs[job_id]
            for key, value in updates.items():
                setattr(job, key, value)

    def _run_job(self, job_id: str, params: dict[str, Any], images: list[Image.Image]) -> None:
        started = time.monotonic()
        self._update_job(
            job_id,
            status="loading_model",
            message="モデルを読み込んでいます",
            started_at=utc_now(),
        )
        try:
            pipe = self.ensure_model()
            self._update_job(job_id, status="running", message="画像を生成しています")

            import torch

            seed = params["seed"]
            if seed < 0:
                seed = secrets.randbelow(2**31)
            generator = torch.Generator(device=CUDA_DEVICE).manual_seed(seed)

            def on_step_end(_pipe: Any, step: int, _timestep: Any, callback_kwargs: dict[str, Any]):
                current = step + 1
                total = params["steps"]
                self._update_job(
                    job_id,
                    step=current,
                    progress=current / total,
                    message=f"生成中 {current} / {total} ステップ",
                )
                return callback_kwargs

            call_args: dict[str, Any] = {
                "prompt": params["prompt"],
                "height": params["height"],
                "width": params["width"],
                "num_inference_steps": params["steps"],
                "true_cfg_scale": params["guidance_scale"],
                "generator": generator,
                "callback_on_step_end": on_step_end,
            }
            if ACCELERATION == "viggle-r128":
                call_args["sigmas"] = VIGGLE_SIGMAS
            if params["guidance_scale"] > 1:
                call_args["negative_prompt"] = params["negative_prompt"] or ""
            if images:
                call_args["image"] = images[0] if len(images) == 1 else images
                call_args["output_resolution"] = reference_resolution(len(images))

            result = pipe(**call_args).images[0]
            filename = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{job_id[:8]}.png"
            output_path = OUTPUT_DIR / filename
            result.save(output_path)

            metadata = {
                **params,
                "seed": seed,
                "reference_images": len(images),
                "reference_resolution": reference_resolution(len(images)) if images else None,
                "quantization": QUANTIZATION,
                "acceleration": ACCELERATION,
                "start_profile": START_PROFILE,
                "cpu_offload": CPU_OFFLOAD,
                "reference_mode": REFERENCE_MODE,
                "vae_tiling": VAE_TILING,
                "trim_cuda_cache": TRIM_CUDA_CACHE,
                "output": filename,
                "elapsed_seconds": round(time.monotonic() - started, 2),
                "created_at": utc_now(),
            }
            metadata_path = output_path.with_suffix(".json")
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

            self._update_job(
                job_id,
                status="completed",
                progress=1.0,
                step=params["steps"],
                message="完了",
                finished_at=utc_now(),
                output_url=f"/outputs/{filename}",
                metadata_url=f"/outputs/{metadata_path.name}",
            )
        except Exception as exc:
            traceback.print_exc()
            self._update_job(
                job_id,
                status="failed",
                message="生成に失敗しました",
                error=str(exc),
                finished_at=utc_now(),
            )
        finally:
            for image in images:
                image.close()
            gc.collect()

    def _prune_jobs_locked(self) -> None:
        if len(self.jobs) <= 100:
            return
        finished = [
            job_id
            for job_id, job in self.jobs.items()
            if job.status in {"completed", "failed"}
        ]
        for job_id in finished[: len(self.jobs) - 100]:
            self.jobs.pop(job_id, None)


service = GenerationService()
