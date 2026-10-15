# Shuji-Tool

**A self-hosted speech-to-text service with an OpenAI-compatible API, stateless realtime decoding, standalone VAD, and speaker diarization.**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)

繁體中文說明：[README.zh-TW.md](README.zh-TW.md)

Shuji-Tool wraps [faster-whisper](https://github.com/SYSTRAN/faster-whisper) in a FastAPI service and adds the pieces you end up needing in production: a drop-in OpenAI transcription endpoint, per-request realtime endpoints that hold no session state, speaker diarization surfaced through structured fields, and post-processing that removes the hallucinated boilerplate Whisper emits on silence.

---

## Why another Whisper server

**Realtime decoding without sticky sessions.** Chunk decoding and voice-activity detection are separate stateless endpoints: every request carries its own complete audio window, and the service keeps no per-stream state. Replicas sit behind a plain round-robin load balancer — no sticky routing, no state blobs in the API contract, no separate deployment for VAD. Windowing and finalization stay with the caller.

**Diarization that survives an OpenAI-shaped API.** Speaker labels are never mixed into transcript text. They appear only in a non-standard `speaker` field on `verbose_json` segments, so downstream consumers such as LiteLLM still see clean transcripts in `text` while retaining the material to reassemble speaker turns.

**Hallucination filtering built in.** Whisper reliably invents subtitle credits and channel sign-offs over silence. Shuji-Tool drops known boilerplate by blacklist and trims leading/trailing segments using `no_speech_prob`, `avg_logprob`, and duration thresholds — all tunable by environment variable.

**A parameter channel that survives a proxy.** faster-whisper-specific knobs (beam size, hotwords, VAD thresholds, diarization) travel as non-standard multipart fields, which OpenAI clients send via `extra_body` and LiteLLM forwards untouched. You keep cost tracking and fallback instead of resorting to a pass-through endpoint.

---

## Quick start

### Docker

```bash
git clone https://github.com/aiDAPTIV-Phison/shuji-tool.git shuji-tool && cd shuji-tool

# Build the plain-Python runtime image (default stage)
make image

docker run --rm --gpus all -p 7766:7766 \
  -v /path/to/models/whisper:/model/whisper:ro \
  shuji-tool/shuji-tool:0.1.0
```

Then:

```bash
curl -s localhost:7766/health
curl -s localhost:7766/v1/audio/transcriptions \
  -F file=@sample.wav \
  -F response_format=verbose_json
```

To enable diarization, mount the extra models and switch it on:

```bash
docker run --rm --gpus all -p 7766:7766 \
  -e ENABLE_DIARIZATION=true \
  -v /path/to/models/whisper:/model/whisper:ro \
  -v /path/to/models/alignment:/model/alignment:ro \
  -v /path/to/models/diarization:/model/diarization:ro \
  shuji-tool/shuji-tool:0.1.0
```

### From source

Requires Python 3.12, FFmpeg on `PATH`, and — for GPU inference — CUDA 12 with cuDNN.

```bash
make init                                    # installs requirements-dev.txt (uses uv by default)
export WHISPER_MODEL_FOLDER=/path/to/models/whisper
make serve                                   # http://0.0.0.0:7766
```

`RUN_IN_UV=0 make init` uses plain `pip` instead of `uv`.

CPU-only hosts work without changes: the service detects CUDA at startup and falls back to `int8` on CPU.

---

## Models

No model weights are included in this repository or in any image built from it. You download them yourself, and the model authors' terms apply to you.

| Purpose | Suggested model | Mounted at | License |
|---|---|---|---|
| Transcription | [`Systran/faster-whisper-large-v3-turbo-ct2`](https://huggingface.co/Systran/faster-whisper-large-v3-turbo-ct2) | `WHISPER_MODEL_FOLDER` | MIT |
| Word alignment | wav2vec2 model for your language, as selected by WhisperX | `ALIGNMENT_MODEL_FOLDER` | Apache-2.0 / MIT |
| Diarization | [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1) | `DIARIZATION_MODEL_FOLDER` | CC-BY-4.0, **gated** |

The transcription model must be in CTranslate2 format. Any CTranslate2-converted Whisper checkpoint works — use a smaller one if `large-v3-turbo` does not fit your GPU.

> **Diarization models are gated.** pyannote pipelines require a Hugging Face account and acceptance of the model's conditions before download. `speaker-diarization-community-1` is CC-BY-4.0: commercial use is permitted, **attribution is required**. Do not commit these weights or bake them into a public image — that would bypass the gate and strip the attribution. See [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).

Alignment and diarization models are only loaded when `ENABLE_DIARIZATION=true`. With diarization off, the service starts without them and rejects any request that asks for speaker labels.

---

## API

### OpenAI-compatible endpoints

#### `POST /v1/audio/transcriptions`

Standard OpenAI fields (`multipart/form-data`):

| Field | Type | Default | Notes |
|---|---|---|---|
| `file` | file | required | Audio to transcribe |
| `model` | string | – | Echoed back only; this service loads one set of weights and does not select models |
| `language` | string | auto | Omitted or empty means auto-detect |
| `prompt` | string | – | Maps to faster-whisper's `initial_prompt` |
| `response_format` | string | `json` | `json`, `verbose_json`, or `text` |
| `timestamp_granularities[]` | string[] | – | Including `word` produces word-level timestamps |

Extended fields, sent through `extra_body` by OpenAI clients:

`beam_size`, `enable_vad`, `use_fast_pipeline`, `condition_on_previous_text`, `repetition_penalty`, `hotwords`, `word_timestamps`, `diarization`, `min_speaker`, `max_speaker`, `always_speaker_label`, `vad_filter_ms`, `vad_threshold`, `neg_threshold`, `min_speech_duration_ms`, `max_speech_duration_s`, `speech_pad_ms`

Anything you omit falls back to the environment-driven defaults. An `X-Request-Id` header, if present, is used as the correlation id in logs.

**Deliberately unsupported.** `temperature` is accepted and discarded with a warning: faster-whisper's default is an ascending temperature sequence used as a fallback when decoding quality drops, and forwarding OpenAI's default of `0` would disable that mechanism and increase hallucination. `stream=true` returns 400.

```bash
curl localhost:7766/v1/audio/transcriptions \
  -F file=@meeting.wav \
  -F response_format=verbose_json \
  -F beam_size=3 \
  -F 'hotwords=Shuji-Tool,CTranslate2' \
  -F diarization=true \
  -F max_speaker=4
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:7766/v1", api_key="not-used")

client.audio.transcriptions.create(
    model="whisper-1",
    file=open("meeting.wav", "rb"),
    response_format="verbose_json",
    extra_body={"beam_size": 5, "diarization": True, "max_speaker": 4},
)
```

`verbose_json` responses carry `task`, `language`, `duration`, `text`, and `segments[]`. `duration` is the source audio's length rather than the sum of segment durations, because hallucination trimming removes segments and a sum would understate the length — which matters when a proxy prices requests by duration.

Speaker labels appear as `segments[].speaker` (or `segments[].words[].speaker` with word-level timestamps), formatted as `SPEAKER_00`, `SPEAKER_01`, … or `UNKNOWN`. `text` never contains them.

#### `GET /v1/models`

Returns a single `whisper-1` alias, since the service loads exactly one set of weights. Present so OpenAI clients and LiteLLM can probe the service.

#### Realtime endpoints

| Endpoint | Purpose |
|---|---|
| `POST /v1/audio/chunk_decode` | Stateless decode of one WAV window (multipart) |
| `POST /v1/audio/chunk_decode_final` | Same contract, distinct path so a gateway can route the final chunk separately |
| `POST /v1/audio/vad` | Speech timestamps for one WAV window (multipart) |

These mirror the native endpoints below but use `multipart/form-data`, so they pass through gateways such as Envoy or LiteLLM that expect file uploads.

### Native endpoints

| Endpoint | Purpose |
|---|---|
| `POST /audio/process_audio/` | Batch transcription from a server-side path or a download URL |
| `POST /audio/chunk_decode/` | Stateless decode; the request body is raw RIFF/WAVE bytes |
| `POST /audio/vad/` | Speech timestamps; the request body is raw RIFF/WAVE bytes |
| `POST /audio/check-audio-format/` | Validate format, duration, and metadata before transcribing |
| `GET /health` | Liveness and readiness probe |
| `GET /metrics` | Fixed health payload for dispatchers that poll it |

`/audio/*` endpoints take their parameters as query strings and return `{"detail": ...}` on error. They predate the OpenAI layer and are kept stable for existing callers.

`POST /audio/vad/` returns speech intervals in milliseconds relative to the start of the submitted window — zero intervals means no speech in that window:

```json
{"speech_timestamps": [{"start": 320, "end": 1810}]}
```

### LiteLLM

```yaml
model_list:
  - model_name: shuji-tool
    litellm_params:
      model: openai/whisper-1
      api_base: http://shuji-tool:7766/v1
      api_key: dummy-key
    model_info:
      mode: audio_transcription
```

`api_base` must end in `/v1`; LiteLLM appends `/audio/transcriptions` itself. Shuji-Tool does not authenticate requests, but `api_key` still has to be set because the OpenAI SDK raises client-side without one. If `UVICORN_ROOT_PATH` is set, include that prefix in `api_base`.

---

## Configuration

Every setting is an environment variable. See [.env.example](.env.example) for a copy-paste reference.

### Models

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL_FOLDER` | `/model/whisper/` | CTranslate2 Whisper model directory |
| `ENABLE_DIARIZATION` | `false` | Load alignment and diarization models. Accepts `true`/`1`/`yes`/`on` and their negations |
| `ALIGNMENT_MODEL_FOLDER` | `/model/alignment/` | Requires `ENABLE_DIARIZATION=true` |
| `DIARIZATION_MODEL_FOLDER` | `/model/diarization/` | Requires `ENABLE_DIARIZATION=true` |

### Decoding

| Variable | Default | Description |
|---|---|---|
| `WHISPER_BEAM_SIZE` | `5` | Default beam size when a request omits it |
| `WHISPER_BATCH_SIZE` | `8` | Batch size for the batched inference pipeline |

### Audio limits

| Variable | Default | Description |
|---|---|---|
| `VTT_SUPPORTED_AUDIO_LENGTHS` | `10800` | Maximum duration in seconds (3 hours) |
| `SUPPORTED_FORMATS` | `.m4a,.mp3,.wav` | Comma-separated accepted extensions |

### Service

| Variable | Default | Description |
|---|---|---|
| `APP_PORT` | `7766` | Listen port |
| `APP_WORKERS` | `1` | Uvicorn workers. Each worker loads its own copy of the model — scale with replicas instead |
| `UVICORN_ROOT_PATH` | – | Public path prefix behind a reverse proxy |
| `LOG_LEVEL` | `INFO` | Log level |

### Hallucination trimming

| Variable | Default | Description |
|---|---|---|
| `HARD_NO_SPEECH_PROB_THRESHOLD` | `0.99` | Above this, the leading or trailing segment is dropped unconditionally |
| `REMOVE_FIRST_SEGMENT_NO_SPEECH_PROB` | `0.55` | Leading-segment threshold, combined with the one below |
| `REMOVE_FIRST_SEGMENT_LOG_PROB` | `-1.2` | Leading segment is dropped when `avg_logprob` is at or below this |
| `REMOVE_LAST_SEGMENT_DURATION` | `2.5` | Only trailing segments shorter than this are candidates |
| `REMOVE_LAST_SEGMENT_NO_SPEECH_PROB` | `0.8` | Trailing-segment threshold |

Trimming applies to batch transcription. Realtime chunk decoding skips it — a window's edges are not a recording's edges — but still applies the blacklist.

### Observability

Shuji-Tool uses the standard OpenTelemetry environment variables. Telemetry is **off** unless `OTEL_SDK_DISABLED` is explicitly set to `false`.

| Variable | Default | Description |
|---|---|---|
| `OTEL_SDK_DISABLED` | `true` | Set to `false` to enable traces, metrics, and logs |
| `OTEL_SERVICE_NAME` | – | Service name in the resource |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | – | Shared OTLP endpoint |
| `OTEL_EXPORTER_OTLP_{TRACES,METRICS,LOGS}_ENDPOINT` | – | Per-signal override |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `grpc` | Only `grpc` is supported; other values log a warning and still use gRPC |

Every request carries a `flow_id` (from `X-Request-Id` on `/v1`, or an explicit parameter on `/audio/*`) that appears in every log line it produces, so a single transcription is greppable end to end.

---

## Transcript post-processing

1. **Hallucination blacklist.** Segments containing known Whisper boilerplate (subtitle credits, channel sign-offs) are dropped entirely.
2. **Edge trimming.** Leading and trailing segments are dropped when their decoding statistics indicate silence, per the thresholds above.
3. **Simplified → Traditional Chinese.** All output passes through OpenCC `s2tw`. Conversion is applied per segment and then concatenated, which guarantees that `text` equals the concatenation of `segments[].text`. Word-level output is converted per word to preserve timestamps, so concatenated `words` may differ slightly from `text`.

Step 3 is unconditional. If you do not want Traditional Chinese output, remove the `OpenCC` conversion in [`src/transcription/service.py`](src/transcription/service.py) and [`src/api/openai_schemas.py`](src/api/openai_schemas.py).

---

## Architecture

```
src/
├── api/
│   ├── app.py                  # FastAPI entry point: routers, telemetry, lifespan
│   ├── openai_router.py        # /v1 endpoints
│   ├── openai_schemas.py       # verbose_json construction, per-segment OpenCC
│   ├── openai_errors.py        # OpenAI error envelope, scoped to /v1
│   ├── process_router.py       # /audio/process_audio/
│   ├── chunk_decode_router.py  # /audio/chunk_decode/
│   ├── vad_router.py           # /audio/vad/
│   ├── checker_router.py       # /audio/check-audio-format/
│   ├── checker.py              # Format, duration, and metadata validation
│   └── dependencies.py         # Engine injection
├── transcription/
│   ├── engine_loader.py        # Startup load, engine cache
│   ├── engine.py               # FasterWhisperEngine
│   ├── async_faster_whisper.py # asyncio.to_thread wrapper
│   ├── service.py              # Post-processing, hallucination filtering, formatting
│   ├── diarization.py          # WhisperX alignment + pyannote, speaker blocks
│   ├── parameters.py           # WhisperParams / DiarizationParams and defaults
│   └── cuda_exception.py       # CUDA OOM classification
├── vad/
│   ├── engine.py               # silero-vad, CPU ONNX, loaded once at startup
│   └── parameters.py
└── utils/                      # Config, logging, audio files, downloads, telemetry
```

Engines are built once during the FastAPI lifespan startup; a failure there fails startup rather than being swallowed. Transcription runs in a thread pool via `asyncio.to_thread()` so the event loop stays responsive, and the realtime VAD engine runs on CPU so it never contends with Whisper for the GPU.

Design decisions and the options rejected along the way are recorded in [`docs/adr/`](docs/adr/). The domain vocabulary — what "speaker block" or "speech timestamps" mean precisely, and which terms to avoid — is in [`CONTEXT.md`](CONTEXT.md).

---

## Error codes

`/v1` endpoints return an OpenAI error envelope with the code preserved in `error.code`:

```json
{"error": {"message": "...", "type": "invalid_request_error", "param": "file", "code": "MSS-VTT-000701"}}
```

`/audio/*` endpoints return `{"detail": {"errorCode": ...}}`.

| Code | Meaning | HTTP |
|---|---|---|
| `MSS-VTT-000000` | Unexpected internal error | 500 |
| `MSS-VTT-000100` | Duration over the limit, unsupported format, or unsupported language | 400 / 500 |
| `MSS-VTT-000101` | Missing required input | 400 |
| `MSS-VTT-000102` | Unsupported parameter value | 400 |
| `MSS-VTT-000300` | File not found | 404 |
| `MSS-VTT-000400` | Download failed | varies |
| `MSS-VTT-000701` | Invalid or empty audio content | 400 / 500 |
| `MSS-VTT-000703` | Path does not exist | 404 |
| `MSS-VTT-500100` | Unexpected CUDA error | 500 |
| `MSS-VTT-500101` | CUDA out of memory | 500 |

---

## Development

```bash
make init     # install dependencies
make test     # pytest
make lint     # pylint
make help     # list targets and variables
```

The test suite stubs out every GPU and native dependency in [`tests/conftest.py`](tests/conftest.py), so it runs on any machine without CUDA, PyTorch, or model weights. `requirements-test.txt` holds that lightweight set; CI installs it.

### Docker build stages

| Stage | Contents | Build with |
|---|---|---|
| `runtime` | Plain Python, non-root — the default | `make image` |
| `dev` | Build tooling plus test dependencies | `STAGE=dev make image` |
| `final` | Nuitka-compiled binary, non-root | `STAGE=final make image` |
| `base` | Shared dependency layer | – |

Override the image coordinates with `IMAGE_REGISTRY`, `IMAGE_NAME`, and `IMAGE_VERSION`; `make push` builds and pushes. The container image defaults to `TZ=UTC` — pass `--build-arg TZ=Asia/Taipei` (or your zone) if log timestamps should be local.

---

## Security

This open-source release does not provide real-time vulnerability fixes; security patches follow the supported versions and release cycle published by this project.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

Third-party components and their licenses are listed in [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md). The Python packages are under permissive licenses. GPU inference relies on NVIDIA CUDA libraries under NVIDIA's proprietary license, which you download yourself when installing; this project does not redistribute them. Model weights are licensed separately by their authors and are not distributed with this project — the diarization pipeline in particular is gated and requires attribution.
