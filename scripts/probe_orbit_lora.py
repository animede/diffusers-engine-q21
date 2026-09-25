#!/usr/bin/env python3
"""Probe Qwen Image 2.1 Orbit LoRA compatibility, speed, and VRAM."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
ORBIT_REPO = "ML-Intern-lab/Qwen-Image-2.1-viewpoint-orbit-LoRA"
ORBIT_REVISION = "b5283f45e9291147ea428c5360445d6cee4d9d63"
ORBIT_RELATIVE_PATH = Path(
    "checkpoints/steps2000res768/orbit_alpha_lora_gate_up_split.safetensors"
)
ORBIT_SHA256 = "bcea69afef59fe0d0126ec865590afb9052f9607e1e8c7898242fe816aac87f9"
VIGGLE_SIGMAS = [1.0, 0.9375, 0.875, 0.75, 0.5, 0.25]
SWEEP_PROMPTS = (
    (
        "left-45",
        "<orbit> rotate the camera 45 degrees to the left, eye level. "
        "The image has alpha channel and the background is transparent.",
    ),
    (
        "left-90",
        "<orbit> rotate the camera 90 degrees to the left, eye level. "
        "The image has alpha channel and the background is transparent.",
    ),
    (
        "left-135",
        "<orbit> rotate the camera 135 degrees to the left, eye level. "
        "The image has alpha channel and the background is transparent.",
    ),
    (
        "back-180",
        "<orbit> rotate the camera 180 degrees, eye level. "
        "The image has alpha channel and the background is transparent.",
    ),
    (
        "right-135",
        "<orbit> rotate the camera 135 degrees to the right, eye level. "
        "The image has alpha channel and the background is transparent.",
    ),
    (
        "right-90",
        "<orbit> rotate the camera 90 degrees to the right, eye level. "
        "The image has alpha channel and the background is transparent.",
    ),
    (
        "right-45",
        "<orbit> rotate the camera 45 degrees to the right, eye level. "
        "The image has alpha channel and the background is transparent.",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--prompt")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "orbit-probe")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--quantization", choices=("nvfp4", "fp8", "bf16"), default="nvfp4")
    parser.add_argument("--acceleration", choices=("none", "viggle-r128"), default="none")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--input-mode", choices=("keep", "rgb", "rgba"), default="keep")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--orbit-weight", type=float, default=1.0)
    parser.add_argument(
        "--skip-orbit",
        action="store_true",
        help="Generate without loading Orbit LoRA",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Generate all seven trained eye-level relative azimuths from the original input",
    )
    parser.add_argument("--label", default="probe")
    return parser.parse_args()


def configure_environment(args: argparse.Namespace) -> None:
    os.environ["QWEN_DEVICE"] = args.device
    os.environ["QWEN_QUANTIZATION"] = args.quantization
    os.environ["QWEN_ACCELERATION"] = args.acceleration
    os.environ["QWEN_CPU_OFFLOAD"] = "1" if args.cpu_offload else "0"
    os.environ["QWEN_MODEL_DIR"] = str(PROJECT_ROOT / "models" / "Qwen-Image-2.1")
    os.environ["QWEN_NVFP4_CACHE_DIR"] = str(PROJECT_ROOT / "models" / "Qwen-Image-2.1-nvfp4")
    os.environ["QWEN_VIGGLE_LORA_DIR"] = str(
        PROJECT_ROOT / "models" / "Qwen-Image-2.1-viggle-turbo"
    )
    os.environ["QWEN_OUTPUT_DIR"] = str(PROJECT_ROOT / "outputs")


def main() -> int:
    args = parse_args()
    if not args.sweep and not args.prompt:
        raise SystemExit("--prompt is required unless --sweep is used")
    if args.sweep and not args.input:
        raise SystemExit("--sweep requires --input")
    configure_environment(args)

    import torch

    device_index = int(args.device.rsplit(":", 1)[1]) if ":" in args.device else 0
    torch.cuda.set_device(device_index)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)

    orbit_path = (
        PROJECT_ROOT
        / "models"
        / "Qwen-Image-2.1-viewpoint-orbit-LoRA"
        / ORBIT_RELATIVE_PATH
    )
    if not orbit_path.is_file():
        raise SystemExit(f"Orbit LoRA is missing: {orbit_path}")
    orbit_sha256 = sha256(orbit_path)
    if orbit_sha256 != ORBIT_SHA256:
        raise SystemExit(
            f"Orbit LoRA SHA-256 mismatch: expected {ORBIT_SHA256}, "
            f"got {orbit_sha256}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = f"{timestamp}-{args.label}-{args.quantization}-{args.acceleration}-{args.input_mode}"
    report_path = args.output_dir / f"{stem}.json"

    report: dict = {
        "created_at": datetime.now(UTC).isoformat(),
        "status": "running",
        "label": args.label,
        "device": args.device,
        "gpu": torch.cuda.get_device_name(device_index),
        "gpu_total_gib": round(
            torch.cuda.get_device_properties(device_index).total_memory / 1024**3, 3
        ),
        "quantization": args.quantization,
        "acceleration": args.acceleration,
        "cpu_offload": args.cpu_offload,
        "orbit_repo": ORBIT_REPO,
        "orbit_revision": ORBIT_REVISION,
        "orbit_sha256": orbit_sha256,
        "orbit_weight": args.orbit_weight,
        "orbit_enabled": not args.skip_orbit,
        "input": str(args.input.resolve()) if args.input else None,
        "input_mode": args.input_mode,
        "prompt": args.prompt,
        "width": args.width,
        "height": args.height,
        "seed": args.seed,
    }

    try:
        from backend.app.service import service

        started = time.perf_counter()
        pipe = service.ensure_model()
        torch.cuda.synchronize(device_index)
        report["model_load_seconds"] = round(time.perf_counter() - started, 3)
        report["allocated_after_model_gib"] = round(
            torch.cuda.memory_allocated(device_index) / 1024**3, 3
        )

        if not args.skip_orbit:
            started = time.perf_counter()
            pipe.load_lora_weights(str(orbit_path), adapter_name="orbit")
            if args.acceleration == "viggle-r128":
                pipe.set_adapters(
                    ["viggle-turbo", "orbit"],
                    adapter_weights=[1.0, args.orbit_weight],
                )
            else:
                pipe.set_adapters(["orbit"], adapter_weights=[args.orbit_weight])
            torch.cuda.synchronize(device_index)
            report["orbit_load_seconds"] = round(time.perf_counter() - started, 3)
            report["allocated_after_orbit_gib"] = round(
                torch.cuda.memory_allocated(device_index) / 1024**3, 3
            )

        source = None
        if args.input:
            source = Image.open(args.input)
            source.load()
            if args.input_mode == "rgb":
                source = source.convert("RGB")
            elif args.input_mode == "rgba":
                source = source.convert("RGBA")
            report["source_mode"] = source.mode
            report["source_size"] = list(source.size)

        steps = args.steps
        if steps is None:
            steps = 6 if args.acceleration == "viggle-r128" else 40
        report["steps"] = steps
        call_args = {
            "width": args.width,
            "height": args.height,
            "output_resolution": min(args.width, args.height),
            "num_inference_steps": steps,
            "true_cfg_scale": 1.0,
            "generator": torch.Generator(device=args.device).manual_seed(args.seed),
        }
        if source is not None:
            call_args["image"] = source
        if args.acceleration == "viggle-r128":
            call_args["sigmas"] = VIGGLE_SIGMAS

        prompts = SWEEP_PROMPTS if args.sweep else (("single", args.prompt),)
        runs = []
        for angle_label, prompt in prompts:
            output_suffix = f"-{angle_label}" if args.sweep else ""
            output_path = args.output_dir / f"{stem}{output_suffix}.png"
            current_call_args = dict(call_args)
            current_call_args["prompt"] = prompt
            current_call_args["generator"] = torch.Generator(
                device=args.device
            ).manual_seed(args.seed)

            torch.cuda.reset_peak_memory_stats(device_index)
            started = time.perf_counter()
            with torch.inference_mode():
                result = pipe(**current_call_args).images[0]
            torch.cuda.synchronize(device_index)
            run = {
                "angle": angle_label,
                "prompt": prompt,
                "generation_seconds": round(time.perf_counter() - started, 3),
                "generation_peak_allocated_gib": round(
                    torch.cuda.max_memory_allocated(device_index) / 1024**3, 3
                ),
                "generation_peak_reserved_gib": round(
                    torch.cuda.max_memory_reserved(device_index) / 1024**3, 3
                ),
                "output_mode": result.mode,
                "output_size": list(result.size),
                "output": str(output_path.resolve()),
            }
            result.save(output_path)
            runs.append(run)

        if args.sweep:
            report["runs"] = runs
            report["generation_total_seconds"] = round(
                sum(run["generation_seconds"] for run in runs), 3
            )
            report["generation_peak_allocated_gib"] = max(
                run["generation_peak_allocated_gib"] for run in runs
            )
            report["generation_peak_reserved_gib"] = max(
                run["generation_peak_reserved_gib"] for run in runs
            )
        else:
            report.update(runs[0])
        report["status"] = "completed"
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        traceback.print_exc()
    finally:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))

    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
