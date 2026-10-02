"""現行TorchAO NVFP4パスのベースライン計測 (bench_nunchaku.pyとの比較用).

実行: .venv/bin/python scripts/bench_torchao_baseline.py --device cuda:1
"""
import argparse
import gc
import json
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "Qwen-Image-2.1"
NVFP4_CACHE_DIR = ROOT / "models" / "Qwen-Image-2.1-nvfp4"
PROMPT = "A serene Japanese garden with a red wooden bridge over a koi pond, autumn maple leaves, soft morning light, photorealistic"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--viggle", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--te-diet", action="store_true")
    args = parser.parse_args()

    import mslk  # noqa: F401
    from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel
    from transformers import Qwen3VLForConditionalGeneration

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    assert (NVFP4_CACHE_DIR / ".complete").is_file(), "NVFP4キャッシュがありません"

    t0 = time.perf_counter()
    transformer = QwenImage21Transformer2DModel.from_pretrained(
        NVFP4_CACHE_DIR / "transformer", dtype=torch.bfloat16,
        local_files_only=True, device_map=str(device),
    )
    gc.collect(); torch.cuda.empty_cache()
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        NVFP4_CACHE_DIR / "text_encoder", dtype=torch.bfloat16,
        local_files_only=True, device_map=str(device),
    )
    gc.collect(); torch.cuda.empty_cache()
    pipe = QwenImage21Pipeline.from_pretrained(
        str(MODEL_DIR), transformer=transformer, text_encoder=text_encoder,
        torch_dtype=torch.bfloat16, local_files_only=True,
    )
    pipe.vae.to(device)
    pipe.vae.enable_tiling()

    if args.te_diet:
        import sys

        sys.path.insert(0, str(ROOT / "scripts"))
        from te_lowvram import apply_te_diet

        freed = apply_te_diet(pipe.text_encoder)
        gc.collect(); torch.cuda.empty_cache()
        print(f"te-diet freed {freed:.2f} GiB")

    if args.compile:
        torch._dynamo.config.cache_size_limit = 256
        pipe.transformer.compile_repeated_blocks(mode=args.compile_mode, dynamic=False)
        print(f"regional compile enabled (mode={args.compile_mode})")

    sigmas = None
    if args.viggle:
        from diffusers import FlowMatchEulerDiscreteScheduler

        viggle_dir = ROOT / "models" / "Qwen-Image-2.1-viggle-turbo"
        lora_file = next(viggle_dir.glob("*.safetensors"))
        pipe.load_lora_weights(str(lora_file), adapter_name="viggle-turbo")
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            str(viggle_dir), subfolder="scheduler"
        )
        args.steps = 6
        sigmas = [1.0, 0.9375, 0.875, 0.75, 0.5, 0.25]
        print("viggle lora loaded (unfused)")
    load_s = time.perf_counter() - t0
    print(f"load: {load_s:.1f}s  resident: {torch.cuda.memory_allocated(device)/2**30:.2f} GiB")

    results = []
    for i in range(args.runs + 1):
        torch.cuda.reset_peak_memory_stats(device)
        gen = torch.Generator(device=device).manual_seed(args.seed)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call_args = dict(
            prompt=PROMPT, num_inference_steps=args.steps,
            width=args.size, height=args.size, generator=gen,
        )
        if args.viggle:
            call_args["sigmas"] = sigmas
            call_args["true_cfg_scale"] = 1.0
        image = pipe(**call_args).images[0]
        end.record()
        torch.cuda.synchronize(device)
        elapsed = start.elapsed_time(end) / 1000
        peak = torch.cuda.max_memory_allocated(device) / 2**30
        tag = "warmup" if i == 0 else f"run{i}"
        print(f"{tag}: {elapsed:.2f}s  peak {peak:.2f} GiB")
        if i > 0:
            results.append({"time_s": elapsed, "peak_gib": peak})
        if i == 1:
            out = ROOT / "outputs" / "bench_torchao_sample.png"
            image.save(out)
            print(f"sample saved: {out}")

    avg = sum(r["time_s"] for r in results) / len(results)
    summary = {
        "path": "torchao-nvfp4-cache", "device": args.device,
        "steps": args.steps, "size": args.size, "avg_time_s": round(avg, 2),
        "peak_gib": round(max(r["peak_gib"] for r in results), 2),
        "load_s": round(load_s, 1), "runs": results,
    }
    print(json.dumps(summary, indent=2))
    (ROOT / "benchmarks" / "torchao_baseline_bench.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
