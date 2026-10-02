#!/usr/bin/env bash

# Shared startup-profile parser. This file is sourced by start.sh and start_backend.sh.

qwen_print_profiles() {
  cat <<'EOF'
Available profiles:
  turbo         Viggle r128 6-step + NVFP4 (fastest, 0-3 references)
  turbo-fp8     Viggle r128 6-step + FP8, GPU resident
  turbo-bf16    Viggle r128 6-step + BF16, GPU resident
  turbo-minimum-vram  Viggle r128 6-step + FP8 CPU offload
  fast          NVFP4 W4A4, GPU resident, adaptive references (Blackwell recommended)
  compatible    FP8 W8A8, GPU resident (higher VRAM, wider fallback)
  quality       BF16 original weights (about 31.6 GiB at 1024px)
  minimum-vram  FP8 with CPU offload (slowest, requires ample system RAM)
EOF
}

qwen_start_usage() {
  cat <<'EOF'
Usage: ./start.sh [options]
       ./start_backend.sh [options]

Options:
  -p, --profile NAME          turbo|turbo-compile|fast|compatible|quality|minimum-vram
                              turbo-fp8|turbo-bf16|turbo-minimum-vram
  -d, --device DEVICE         CUDA device, for example cuda:0
      --quantization MODE     nvfp4|fp8|bf16 (advanced override)
      --acceleration MODE     none|viggle-r128 (advanced override)
      --cpu-offload           Enable CPU offload
      --no-cpu-offload        Disable CPU offload
      --reference-mode MODE   adaptive|full
      --vae-tiling            Enable tiled VAE decoding
      --no-vae-tiling         Disable tiled VAE decoding
      --trim-cache            Release temporary CUDA cache between stages
      --no-trim-cache         Keep the CUDA allocator cache
      --te-diet               Move text-encoder embeddings/LM head to CPU (tediet)
      --te-stream             Stream text-encoder layers from pinned host (tediet)
      --compile-blocks        Regional-compile the DiT blocks (first job compiles)
      --non-interactive       Do not show the startup menu
      --list-profiles         Show profiles and exit
  -h, --help                  Show this help

With no profile or QWEN_QUANTIZATION environment variable, an interactive
terminal shows a menu. Non-interactive launches default to the fast profile.
EOF
}

qwen_apply_profile() {
  case "$1" in
    fast|nvfp4)
      QWEN_START_PROFILE="fast"
      QWEN_QUANTIZATION="nvfp4"
      QWEN_CPU_OFFLOAD="0"
      QWEN_REFERENCE_MODE="adaptive"
      QWEN_VAE_TILING="1"
      QWEN_TRIM_CUDA_CACHE="1"
      QWEN_ACCELERATION="none"
      ;;
    turbo-compile)
      QWEN_START_PROFILE="turbo-compile"
      QWEN_QUANTIZATION="nvfp4"
      QWEN_CPU_OFFLOAD="0"
      QWEN_REFERENCE_MODE="adaptive"
      QWEN_VAE_TILING="1"
      QWEN_TRIM_CUDA_CACHE="1"
      QWEN_ACCELERATION="viggle-r128"
      QWEN_TE_DIET="1"
      QWEN_COMPILE_BLOCKS="1"
      ;;
    turbo|viggle)
      QWEN_START_PROFILE="turbo"
      QWEN_QUANTIZATION="nvfp4"
      QWEN_CPU_OFFLOAD="0"
      QWEN_REFERENCE_MODE="adaptive"
      QWEN_VAE_TILING="1"
      QWEN_TRIM_CUDA_CACHE="1"
      QWEN_ACCELERATION="viggle-r128"
      ;;
    turbo-fp8)
      QWEN_START_PROFILE="turbo-fp8"
      QWEN_QUANTIZATION="fp8"
      QWEN_CPU_OFFLOAD="0"
      QWEN_REFERENCE_MODE="adaptive"
      QWEN_VAE_TILING="1"
      QWEN_TRIM_CUDA_CACHE="1"
      QWEN_ACCELERATION="viggle-r128"
      ;;
    turbo-bf16)
      QWEN_START_PROFILE="turbo-bf16"
      QWEN_QUANTIZATION="bf16"
      QWEN_CPU_OFFLOAD="0"
      QWEN_REFERENCE_MODE="full"
      QWEN_VAE_TILING="1"
      QWEN_TRIM_CUDA_CACHE="0"
      QWEN_ACCELERATION="viggle-r128"
      ;;
    turbo-minimum-vram)
      QWEN_START_PROFILE="turbo-minimum-vram"
      QWEN_QUANTIZATION="fp8"
      QWEN_CPU_OFFLOAD="1"
      QWEN_REFERENCE_MODE="adaptive"
      QWEN_VAE_TILING="1"
      QWEN_TRIM_CUDA_CACHE="1"
      QWEN_ACCELERATION="viggle-r128"
      ;;
    compatible|fp8)
      QWEN_START_PROFILE="compatible"
      QWEN_QUANTIZATION="fp8"
      QWEN_CPU_OFFLOAD="0"
      QWEN_REFERENCE_MODE="adaptive"
      QWEN_VAE_TILING="1"
      QWEN_TRIM_CUDA_CACHE="1"
      QWEN_ACCELERATION="none"
      ;;
    quality|bf16)
      QWEN_START_PROFILE="quality"
      QWEN_QUANTIZATION="bf16"
      QWEN_CPU_OFFLOAD="0"
      QWEN_REFERENCE_MODE="full"
      QWEN_VAE_TILING="1"
      QWEN_TRIM_CUDA_CACHE="0"
      QWEN_ACCELERATION="none"
      ;;
    minimum-vram|offload)
      QWEN_START_PROFILE="minimum-vram"
      QWEN_QUANTIZATION="fp8"
      QWEN_CPU_OFFLOAD="1"
      QWEN_REFERENCE_MODE="adaptive"
      QWEN_VAE_TILING="1"
      QWEN_TRIM_CUDA_CACHE="1"
      QWEN_ACCELERATION="none"
      ;;
    *)
      echo "Unknown profile: $1" >&2
      qwen_print_profiles >&2
      return 2
      ;;
  esac
}

qwen_choose_profile() {
  {
    echo ""
    echo "Qwen Image 2.1 startup profile"
    echo "  1) Turbo 6-step [recommended for speed]"
    echo "     Viggle r128 + NVFP4, about 3.6 s / 12.6 GiB for 1024px T2I"
    echo "  2) Fast + low VRAM [base 40-step schedule]"
    echo "     NVFP4, about 12.0 s / 11.9 GiB for 1024px T2I"
    echo "  3) FP8 compatibility"
    echo "     FP8, about 18.5 s / 18.5 GiB for 1024px T2I"
    echo "  4) BF16 original weights"
    echo "     About 20.1 s / 31.6 GiB for 1024px T2I"
    echo "  5) Minimum VRAM"
    echo "     FP8 CPU offload; about 27.2 s / 9.8 GiB and uses system RAM"
    echo "  6) Turbo FP8"
    echo "     Viggle 6-step + FP8 GPU resident"
    echo "  7) Turbo BF16"
    echo "     Viggle 6-step + original BF16 weights"
    echo "  8) Turbo Minimum VRAM"
    echo "     Viggle 6-step + FP8 CPU offload"
  } >&2
  read -r -p "Select [1]: " qwen_choice
  case "${qwen_choice:-1}" in
    1) printf '%s' turbo ;;
    2) printf '%s' fast ;;
    3) printf '%s' compatible ;;
    4) printf '%s' quality ;;
    5) printf '%s' minimum-vram ;;
    6) printf '%s' turbo-fp8 ;;
    7) printf '%s' turbo-bf16 ;;
    8) printf '%s' turbo-minimum-vram ;;
    *) echo "Invalid selection: ${qwen_choice}" >&2; return 2 ;;
  esac
}

qwen_configure_startup() {
  local profile="${QWEN_START_PROFILE:-}"
  local device_override=""
  local quantization_override=""
  local cpu_offload_override=""
  local reference_mode_override=""
  local vae_tiling_override=""
  local trim_cache_override=""
  local acceleration_override=""
  local non_interactive="0"

  while (($#)); do
    case "$1" in
      -p|--profile)
        [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; return 2; }
        profile="$2"; shift 2 ;;
      -d|--device)
        [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; return 2; }
        device_override="$2"; shift 2 ;;
      --quantization)
        [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; return 2; }
        quantization_override="$2"; shift 2 ;;
      --cpu-offload) cpu_offload_override="1"; shift ;;
      --no-cpu-offload) cpu_offload_override="0"; shift ;;
      --reference-mode)
        [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; return 2; }
        reference_mode_override="$2"; shift 2 ;;
      --vae-tiling) vae_tiling_override="1"; shift ;;
      --no-vae-tiling) vae_tiling_override="0"; shift ;;
      --trim-cache) trim_cache_override="1"; shift ;;
      --no-trim-cache) trim_cache_override="0"; shift ;;
      --te-diet) QWEN_TE_DIET="1"; shift ;;
      --te-stream) QWEN_TE_STREAM="1"; shift ;;
      --compile-blocks) QWEN_COMPILE_BLOCKS="1"; shift ;;
      --acceleration)
        [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; return 2; }
        acceleration_override="$2"; shift 2 ;;
      --non-interactive) non_interactive="1"; shift ;;
      --list-profiles) qwen_print_profiles; return 10 ;;
      -h|--help) qwen_start_usage; return 10 ;;
      *) echo "Unknown option: $1" >&2; qwen_start_usage >&2; return 2 ;;
    esac
  done

  if [[ -z "$profile" && -z "${QWEN_QUANTIZATION:-}" ]]; then
    if [[ "$non_interactive" == "0" && -t 0 ]]; then
      profile="$(qwen_choose_profile)" || return
    else
      profile="fast"
    fi
  fi
  if [[ -n "$profile" ]]; then
    qwen_apply_profile "$profile" || return
  else
    QWEN_START_PROFILE="custom"
    QWEN_QUANTIZATION="${QWEN_QUANTIZATION:-nvfp4}"
    QWEN_CPU_OFFLOAD="${QWEN_CPU_OFFLOAD:-0}"
    QWEN_REFERENCE_MODE="${QWEN_REFERENCE_MODE:-adaptive}"
    QWEN_VAE_TILING="${QWEN_VAE_TILING:-1}"
    QWEN_TRIM_CUDA_CACHE="${QWEN_TRIM_CUDA_CACHE:-1}"
    QWEN_ACCELERATION="${QWEN_ACCELERATION:-none}"
  fi

  [[ -z "$device_override" ]] || QWEN_DEVICE="$device_override"
  [[ -z "$quantization_override" ]] || { QWEN_QUANTIZATION="$quantization_override"; QWEN_START_PROFILE="custom"; }
  [[ -z "$cpu_offload_override" ]] || { QWEN_CPU_OFFLOAD="$cpu_offload_override"; QWEN_START_PROFILE="custom"; }
  [[ -z "$reference_mode_override" ]] || { QWEN_REFERENCE_MODE="$reference_mode_override"; QWEN_START_PROFILE="custom"; }
  [[ -z "$vae_tiling_override" ]] || { QWEN_VAE_TILING="$vae_tiling_override"; QWEN_START_PROFILE="custom"; }
  [[ -z "$trim_cache_override" ]] || { QWEN_TRIM_CUDA_CACHE="$trim_cache_override"; QWEN_START_PROFILE="custom"; }
  [[ -z "$acceleration_override" ]] || { QWEN_ACCELERATION="$acceleration_override"; QWEN_START_PROFILE="custom"; }
  QWEN_DEVICE="${QWEN_DEVICE:-cuda:0}"

  case "$QWEN_QUANTIZATION" in nvfp4|fp8|bf16) ;; *) echo "Invalid quantization: $QWEN_QUANTIZATION" >&2; return 2 ;; esac
  case "$QWEN_REFERENCE_MODE" in adaptive|full) ;; *) echo "Invalid reference mode: $QWEN_REFERENCE_MODE" >&2; return 2 ;; esac
  case "$QWEN_CPU_OFFLOAD" in 0|1) ;; *) echo "CPU offload must be 0 or 1" >&2; return 2 ;; esac
  case "$QWEN_VAE_TILING" in 0|1) ;; *) echo "VAE tiling must be 0 or 1" >&2; return 2 ;; esac
  case "$QWEN_TRIM_CUDA_CACHE" in 0|1) ;; *) echo "CUDA cache trimming must be 0 or 1" >&2; return 2 ;; esac
  case "$QWEN_ACCELERATION" in none|viggle-r128) ;; *) echo "Acceleration must be none or viggle-r128" >&2; return 2 ;; esac
  if [[ "$QWEN_QUANTIZATION" == "nvfp4" && "$QWEN_CPU_OFFLOAD" == "1" ]]; then
    echo "NVFP4 cannot be combined with CPU offload." >&2
    return 2
  fi

  QWEN_TE_DIET="${QWEN_TE_DIET:-0}"
  QWEN_TE_STREAM="${QWEN_TE_STREAM:-0}"
  QWEN_COMPILE_BLOCKS="${QWEN_COMPILE_BLOCKS:-0}"
  export QWEN_START_PROFILE QWEN_DEVICE QWEN_QUANTIZATION QWEN_CPU_OFFLOAD
  export QWEN_REFERENCE_MODE QWEN_VAE_TILING QWEN_TRIM_CUDA_CACHE
  export QWEN_ACCELERATION QWEN_TE_DIET QWEN_TE_STREAM QWEN_COMPILE_BLOCKS
  export QWEN_OPTIONS_CONFIGURED=1

  echo "Startup: profile=$QWEN_START_PROFILE device=$QWEN_DEVICE quantization=$QWEN_QUANTIZATION acceleration=$QWEN_ACCELERATION cpu_offload=$QWEN_CPU_OFFLOAD reference_mode=$QWEN_REFERENCE_MODE vae_tiling=$QWEN_VAE_TILING trim_cache=$QWEN_TRIM_CUDA_CACHE"
}
