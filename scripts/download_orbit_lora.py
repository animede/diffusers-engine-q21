from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = PROJECT_ROOT / "models" / "Qwen-Image-2.1-viewpoint-orbit-LoRA"
DEFAULT_REVISION = "b5283f45e9291147ea428c5360445d6cee4d9d63"
WEIGHT_RELATIVE_PATH = Path(
    "checkpoints/steps2000res768/orbit_alpha_lora_gate_up_split.safetensors"
)
EXPECTED_SHA256 = "bcea69afef59fe0d0126ec865590afb9052f9607e1e8c7898242fe816aac87f9"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the Qwen Image 2.1 viewpoint Orbit LoRA."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--accept-qwen-research-license",
        action="store_true",
        help="Confirm acceptance of the non-commercial Qwen Research License",
    )
    args = parser.parse_args()
    if not args.accept_qwen_research_license:
        parser.error(
            "read the Qwen Research License and pass --accept-qwen-research-license"
        )

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(
        "Downloading ML-Intern-lab/Qwen-Image-2.1-viewpoint-orbit-LoRA "
        f"to {output}"
    )
    snapshot_download(
        repo_id="ML-Intern-lab/Qwen-Image-2.1-viewpoint-orbit-LoRA",
        revision=args.revision,
        local_dir=output,
        allow_patterns=[
            str(WEIGHT_RELATIVE_PATH),
            "README.md",
            "LICENSE",
        ],
    )

    weight_path = output / WEIGHT_RELATIVE_PATH
    actual_sha256 = sha256(weight_path)
    if args.revision == DEFAULT_REVISION and actual_sha256 != EXPECTED_SHA256:
        raise RuntimeError(
            f"SHA-256 mismatch for {weight_path}: "
            f"expected {EXPECTED_SHA256}, got {actual_sha256}"
        )
    print(f"Download complete: {weight_path}")
    print(f"SHA-256: {actual_sha256}")


if __name__ == "__main__":
    main()
