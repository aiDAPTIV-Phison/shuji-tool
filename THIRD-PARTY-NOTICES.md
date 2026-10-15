# Third-Party Notices

Shuji-Tool is distributed under the Apache License 2.0. The Python packages
listed below are under permissive licenses. Some components that you download
when installing the GPU dependencies or building the Docker image are not: see
[NVIDIA components](#nvidia-components) and
[System components](#system-components).

If you redistribute a built artifact that bundles these packages (for example a
Docker image containing `site-packages`), the BSD- and MIT-licensed components
require that their copyright notices be reproduced with it — this file serves
that purpose.

## Runtime dependencies (code)

| Component | License | Copyright |
|---|---|---|
| [WhisperX](https://github.com/m-bain/whisperX) | BSD-2-Clause | Copyright (c) 2024, Max Bain |
| [pyannote.audio](https://github.com/pyannote/pyannote-audio) | MIT | Copyright (c) CNRS |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | MIT | Copyright (c) 2023 SYSTRAN |
| [CTranslate2](https://github.com/OpenNMT/CTranslate2) | MIT | Copyright (c) 2019-present OpenNMT |
| [silero-vad](https://github.com/snakers4/silero-vad) | MIT | Copyright (c) 2024-present Silero Team |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | MIT | Copyright (c) Microsoft Corporation |
| [FastAPI](https://github.com/fastapi/fastapi) | MIT | Copyright (c) 2018 Sebastián Ramírez |
| [pydub](https://github.com/jiaaro/pydub) | MIT | Copyright (c) 2011 James Robert |
| [python-multipart](https://github.com/Kludex/python-multipart) | Apache-2.0 | Copyright (c) 2012-2013 Andrew Dunham |
| [OpenCC](https://github.com/BYVoid/OpenCC) | Apache-2.0 | Copyright (c) Carbo Kuo and contributors |
| [Transformers](https://github.com/huggingface/transformers) | Apache-2.0 | Copyright (c) Hugging Face |
| [requests](https://github.com/psf/requests) | Apache-2.0 | Copyright (c) Kenneth Reitz |
| [OpenTelemetry Python](https://github.com/open-telemetry/opentelemetry-python) | Apache-2.0 | Copyright (c) The OpenTelemetry Authors |
| [Starlette](https://github.com/encode/starlette) | BSD-3-Clause | Copyright (c) 2018, Encode OSS Ltd |
| [Uvicorn](https://github.com/encode/uvicorn) | BSD-3-Clause | Copyright (c) 2017-present, Encode OSS Ltd |
| [PyTorch](https://github.com/pytorch/pytorch) | BSD-3-Clause | Copyright (c) Meta Platforms, Inc. and affiliates |
| [Pydantic](https://github.com/pydantic/pydantic) | MIT | Copyright (c) 2017 to present Pydantic Services Inc. |

## Model weights (not distributed with this project)

Model weights are loaded at runtime from directories you provide. They are
neither committed to this repository nor baked into any image built from the
included `Dockerfile`. Their own terms apply to you as the person who downloads
them.

| Model | License | Notes |
|---|---|---|
| [`Systran/faster-whisper-large-v3-turbo-ct2`](https://huggingface.co/Systran/faster-whisper-large-v3-turbo-ct2) | MIT | CTranslate2 conversion of OpenAI Whisper (MIT) |
| [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1) | CC-BY-4.0 | **Gated** — requires a Hugging Face account and accepting the model's conditions. Commercial use is allowed; **attribution is required**. |
| [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1) | MIT | **Gated** — the pipeline used by pyannote.audio 3.x |
| wav2vec2 alignment models | Apache-2.0 / MIT | Selected per language by WhisperX; see the model card of the one you use |

## NVIDIA components

GPU inference uses NVIDIA CUDA libraries that are distributed under NVIDIA's
proprietary license, not an open-source license: the `nvidia-*-cu12` wheels
that PyTorch installs (cuBLAS, cuDNN, cuFFT, cuRAND, cuSOLVER, cuSPARSE,
cuSPARSELt, cuFile, nvJitLink, CUDA runtime, NVRTC, CUPTI) and the
`nvidia/cuda` base image in the `Dockerfile`.

This repository contains only source code and the `Dockerfile`; it does not
include or redistribute any NVIDIA software, and no pre-built image is
published. When you `pip install` the requirements or build the image, you
download these components from PyPI or NVIDIA directly and accept NVIDIA's
license terms yourself. If you redistribute an image you built, review those
terms first. CPU-only use does not need them.

## System components

FFmpeg is installed in the container image and is used both as a CLI and, via
its shared `libav*` libraries, by `torchcodec`. Ubuntu's FFmpeg packages are
built with `--enable-gpl`, so they are GPL-licensed builds; they are installed
by the package manager and dynamically linked. If you publish a pre-built
image, verify that the FFmpeg build you ship matches your distribution policy.
