"""Nunchaku SVDQuant NVFP4 検証ベンチ (Qwen-Image 2.1, Blackwell).

既存の models/Qwen-Image-2.1 からテキストエンコーダ/VAEを流用し、
DiTのみ Nunchaku NVFP4 チェックポイントに差し替えて速度/VRAMを測る。

実行: .venv-nunchaku/bin/python scripts/bench_nunchaku.py --device cuda:1
"""
import argparse
import json
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "Qwen-Image-2.1"
NUNCHAKU_CKPT = ROOT / "models" / "Qwen-Image-2.1-nunchaku" / "svdq-fp4_r32-qwen-image-2.1.safetensors"
PROMPT = "A serene Japanese garden with a red wooden bridge over a koi pond, autumn maple leaves, soft morning light, photorealistic"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--te-device", default="cuda:0")
    parser.add_argument("--te-lowvram", action="store_true")
    parser.add_argument("--te-window", type=int, default=2)
    parser.add_argument("--viggle", action="store_true")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from diffusers import QwenImage21Pipeline
    from nunchaku_qwen21 import load_nunchaku_qwen21_transformer

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    t0 = time.perf_counter()
    transformer = load_nunchaku_qwen21_transformer(
        NUNCHAKU_CKPT, MODEL_DIR / "transformer" / "config.json", device=device,
    )
    pipe = QwenImage21Pipeline.from_pretrained(
        str(MODEL_DIR), transformer=None, torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    pipe.transformer = transformer
    te_device = device if args.te_lowvram else torch.device(args.te_device)
    pipe.text_encoder.to(te_device)
    pipe.vae.to(device)

    if args.te_lowvram:
        from te_lowvram import apply_te_diet, apply_te_stream

        freed = apply_te_diet(pipe.text_encoder)
        pinned = apply_te_stream(pipe.text_encoder, device=device, window=args.te_window)
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        print(f"te-lowvram: diet freed {freed:.2f} GiB, streamed(pinned) {pinned:.2f} GiB")

    sigmas = None
    if args.viggle:
        from diffusers import FlowMatchEulerDiscreteScheduler
        from viggle_merge import merge_viggle_lora

        viggle_dir = ROOT / "models" / "Qwen-Image-2.1-viggle-turbo"
        lora_file = next(viggle_dir.glob("*.safetensors"))
        stats = merge_viggle_lora(pipe.transformer, lora_file)
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            str(viggle_dir), subfolder="scheduler"
        )
        args.steps = 6
        sigmas = [1.0, 0.9375, 0.875, 0.75, 0.5, 0.25]
        print(f"viggle merged: {stats}")

    # テキストエンコーダは別デバイスで実行し、埋め込みだけDiT側へ移す
    orig_encode_prompt = pipe.encode_prompt

    def cross_device_encode_prompt(*a, **kw):
        kw.pop("device", None)
        embeds = orig_encode_prompt(*a, device=te_device, **kw)
        moved = tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in embeds)
        torch.cuda.empty_cache()
        return moved

    pipe.encode_prompt = cross_device_encode_prompt

    # _execution_device がTE側(cuda:0)を拾わないようDiT側に固定する
    pipe.__class__ = type(
        "QwenImage21PipelineBench",
        (pipe.__class__,),
        {"_execution_device": property(lambda self: device)},
    )
    pipe.vae.enable_tiling()
    load_s = time.perf_counter() - t0
    print(f"load: {load_s:.1f}s  resident: {torch.cuda.memory_allocated(device)/2**30:.2f} GiB")

    if args.cuda_graph:
        from capture_prep import install_capture_prep
        from cudagraph import ForwardGraphRunner

        install_capture_prep(pipe.transformer)
        runner = ForwardGraphRunner(pipe.transformer)
        runner.install()
        print("cudagraph: prep cache + runner installed")

    results = []
    for i in range(args.runs + 1):  # +1 warmup
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
        if args.cuda_graph:
            call_args["use_kv_cache"] = False
        image = pipe(**call_args).images[0]
        if args.cuda_graph:
            torch.cuda.empty_cache()
        end.record()
        torch.cuda.synchronize(device)
        elapsed = start.elapsed_time(end) / 1000
        peak = torch.cuda.max_memory_allocated(device) / 2**30
        tag = "warmup" if i == 0 else f"run{i}"
        print(f"{tag}: {elapsed:.2f}s  peak {peak:.2f} GiB")
        if i > 0:
            results.append({"time_s": elapsed, "peak_gib": peak})
        if i == 1:
            suffix = "_viggle" if args.viggle else ""
            out = ROOT / "outputs" / f"bench_nunchaku_sample{suffix}.png"
            out.parent.mkdir(exist_ok=True)
            image.save(out)
            print(f"sample saved: {out}")

    if args.cuda_graph:
        print(f"cudagraph: replays={runner.replays} captures={len(runner._captures)}")
    avg = sum(r["time_s"] for r in results) / len(results)
    summary = {
        "checkpoint": NUNCHAKU_CKPT.name, "device": args.device,
        "steps": args.steps, "size": args.size, "avg_time_s": round(avg, 2),
        "peak_gib": round(max(r["peak_gib"] for r in results), 2),
        "load_s": round(load_s, 1), "runs": results,
    }
    print(json.dumps(summary, indent=2))
    (ROOT / "benchmarks" / "nunchaku_bench.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
