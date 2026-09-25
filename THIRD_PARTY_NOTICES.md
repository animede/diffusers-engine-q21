# Third-party notices

The Apache License 2.0 in this repository applies to Diffusers Engine Q21's
original source code and documentation only. It does not replace or modify any
license governing model weights, adapters, libraries, or trademarks.

## Qwen Image 2.1

- Provider: Hangzhou Tongyi Laboratory Technology Co., Ltd.
- Source: <https://huggingface.co/Qwen/Qwen-Image-2.1>
- License: [local copy of the Qwen RESEARCH LICENSE AGREEMENT](licenses/QWEN-RESEARCH-LICENSE.txt) ([upstream](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE))
- Use restriction: non-commercial research or evaluation only unless a separate commercial license is obtained from the licensor.

Required attribution:

> Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) 2026 Hangzhou Tongyi Laboratory Technology Co., Ltd. All Rights Reserved.

The model weights are not included in this Git repository. They are downloaded
directly from the provider only after the user explicitly accepts the model
license in the download command.

## Viggle Qwen Image 2.1 Turbo

- Provider: Viggle
- Source: <https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo>
- License: Qwen RESEARCH LICENSE AGREEMENT
- Relationship: derivative adapter and scheduler for Qwen Image 2.1
- Use restriction: non-commercial research or evaluation only unless a separate commercial license is obtained from the Qwen licensor.

The Viggle weights are not included in this Git repository. The download script
retrieves only the selected adapter, scheduler, model card, license, and NOTICE
from the provider after explicit license acceptance.

## Qwen Image 2.1 Viewpoint Orbit LoRA

- Provider: ML-Intern-lab
- Source: <https://huggingface.co/ML-Intern-lab/Qwen-Image-2.1-viewpoint-orbit-LoRA>
- License: Qwen RESEARCH LICENSE AGREEMENT
- Relationship: derivative viewpoint-control adapter for Qwen Image 2.1
- Use restriction: non-commercial research or evaluation only unless a separate commercial license is obtained from the Qwen licensor.

The Orbit weights are not included in this Git repository. The optional
download script retrieves the selected adapter, model card, and license from
the provider only after explicit license acceptance.

## Software dependencies

Diffusers, Transformers, PyTorch, TorchAO, FastAPI, Uvicorn, Pillow, PEFT,
Accelerate, Hugging Face Hub, Safetensors, MSLK, and their transitive
dependencies remain subject to their respective licenses. Consult the package
metadata and upstream repositories before redistribution.

## Trademarks

Qwen and other third-party names may be trademarks of their respective owners.
Their use in this repository is descriptive and does not imply endorsement.
