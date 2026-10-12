# FAQ

## Q: `make init` or `make serve` fails because Python is too old. What should I do?

- ID: `shuji-tool-install-001`
- Public: Yes
- Applies to: `3.3.*`
- Category: Installation
- Keywords:
  - `Python`
  - `make init`
  - `make serve`
- Approved answer:
  1. `This project requires Python 3.12, FFmpeg on PATH, and — for GPU inference — CUDA 12 with cuDNN. Run python --version, install Python 3.12 if needed, then run make init and make serve. RUN_IN_UV=0 make init uses pip instead of uv.`
- Escalate to GitHub Issues when:
  - Python is already 3.12 and FFmpeg is on PATH, but `make init` still fails
  - The error message does not match a missing interpreter or a missing FFmpeg binary

## Q: The service starts, but transcription fails because no model is loaded. What should I do?

- ID: `shuji-tool-models-001`
- Public: Yes
- Applies to: `3.3.*`
- Category: Models
- Keywords:
  - `WHISPER_MODEL_FOLDER`
  - `model weights`
  - `CTranslate2`
- Approved answer:
  1. `This repository and the images built from it do not include model weights. Download a CTranslate2 Whisper checkpoint yourself and set WHISPER_MODEL_FOLDER to that directory, or mount it at /model/whisper when running the container. A suggested checkpoint is Systran/faster-whisper-large-v3-turbo-ct2. The model authors' terms apply to you.`
- Escalate to GitHub Issues when:
  - `WHISPER_MODEL_FOLDER` points at a CTranslate2 checkpoint and the service still fails at startup
  - The error message does not match a missing or unreadable model directory

## Q: A request with `diarization=true` is rejected. What should I do?

- ID: `shuji-tool-diarization-001`
- Public: Yes
- Applies to: `3.3.*`
- Category: Configuration
- Keywords:
  - `ENABLE_DIARIZATION`
  - `diarization`
  - `pyannote`
- Approved answer:
  1. `Speaker labels are loaded only when ENABLE_DIARIZATION=true. Mount an alignment model and a diarization model, set ALIGNMENT_MODEL_FOLDER and DIARIZATION_MODEL_FOLDER, and start the service with ENABLE_DIARIZATION=true. The suggested diarization pipeline is pyannote/speaker-diarization-community-1, which is gated: accept its conditions on Hugging Face before download, and keep the required CC-BY-4.0 attribution. With diarization left off, the service starts without those models and rejects any request that asks for speaker labels.`
- Escalate to GitHub Issues when:
  - `ENABLE_DIARIZATION=true` and both model directories are mounted, but the request is still rejected
  - The error message does not match diarization being disabled or a missing model directory

## Q: I passed `temperature=0` and the service warned and ignored it. What should I do?

- ID: `shuji-tool-api-001`
- Public: Yes
- Applies to: `3.3.*`
- Category: API
- Keywords:
  - `temperature`
  - `stream`
  - `OpenAI`
- Approved answer:
  1. `temperature is accepted and discarded on purpose. faster-whisper uses an ascending temperature sequence when decoding quality drops, and forwarding OpenAI's default of 0 would disable that fallback. Omit temperature. stream=true is not supported and returns 400. This service also does not authenticate callers; an api_key is only required by the OpenAI client library.`
- Escalate to GitHub Issues when:
  - The request omitted `temperature` and did not set `stream=true`, but the service still returns 400
  - The error message does not match an unsupported parameter
