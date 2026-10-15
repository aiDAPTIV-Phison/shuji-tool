# Shuji-Tool

**自架的語音轉文字服務：OpenAI 相容 API、無狀態即時解碼、獨立 VAD、語者分離。**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)

English: [README.md](README.md)

Shuji-Tool 以 FastAPI 封裝 [faster-whisper](https://github.com/SYSTRAN/faster-whisper)，並補上實際上線後會需要的幾塊：可直接替換的 OpenAI 轉錄端點、不持有 session 狀態的即時端點、以結構化欄位承載的語者分離，以及移除 Whisper 在靜音處捏造內容的後處理。

---

## 為什麼不是又一個 Whisper server

**即時解碼不需要 sticky session。** Chunk 解碼與語音活動偵測是各自獨立的無狀態端點：每次請求自帶完整音訊窗，服務不持有 per-stream 狀態。副本可以掛在單純的 round-robin 負載平衡器後面——不需要 sticky routing、API 契約裡不放 state blob、也不必為 VAD 另開一個 Deployment。切窗與判斷是否 final 由呼叫端負責。

**語者分離不破壞 OpenAI 契約。** Speaker label 不會混進逐字稿文字，只出現在 `verbose_json` 片段的非標準 `speaker` 欄位。這樣 LiteLLM 這類下游拿到的 `text` 仍是乾淨逐字稿，同時保有能重組語者輪次的材料。

**內建幻覺過濾。** Whisper 在靜音處會穩定地捏造字幕組署名與頻道結語。Shuji-Tool 以 blacklist 移除已知樣板，並依 `no_speech_prob`、`avg_logprob` 與片段長度砍掉頭尾片段，門檻全部可由環境變數調整。

**參數通道穿得過 proxy。** faster-whisper 專屬旋鈕（beam size、熱詞、VAD 門檻、diarization）以非標準 multipart 欄位傳遞，OpenAI client 透過 `extra_body` 送出，LiteLLM 原樣轉發。不必改用 pass-through 端點，因此成本追蹤與 fallback 都還在。

---

## 快速開始

### Docker

```bash
git clone https://github.com/aiDAPTIV-Phison/shuji-tool.git shuji-tool && cd shuji-tool

# 建置純 Python 的 runtime image（預設 stage）
make image

docker run --rm --gpus all -p 7766:7766 \
  -v /path/to/models/whisper:/model/whisper:ro \
  shuji-tool/shuji-tool:0.1.0
```

接著：

```bash
curl -s localhost:7766/health
curl -s localhost:7766/v1/audio/transcriptions \
  -F file=@sample.wav \
  -F response_format=verbose_json
```

要開啟語者分離，額外掛上模型並開旗標：

```bash
docker run --rm --gpus all -p 7766:7766 \
  -e ENABLE_DIARIZATION=true \
  -v /path/to/models/whisper:/model/whisper:ro \
  -v /path/to/models/alignment:/model/alignment:ro \
  -v /path/to/models/diarization:/model/diarization:ro \
  shuji-tool/shuji-tool:0.1.0
```

### 從原始碼執行

需要 Python 3.12、`PATH` 上有 FFmpeg；要用 GPU 推理則需 CUDA 12 與 cuDNN。

```bash
make init                                    # 安裝 requirements-dev.txt（預設走 uv）
export WHISPER_MODEL_FOLDER=/path/to/models/whisper
make serve                                   # http://0.0.0.0:7766
```

`RUN_IN_UV=0 make init` 會改用一般的 `pip`。

純 CPU 主機不必改設定：服務啟動時偵測 CUDA，沒有就退回 CPU 的 `int8`。

---

## 模型

本專案的 repo 與由它建出的 image 都不含任何模型權重。權重由你自行下載，並直接適用模型作者的授權條款。

| 用途 | 建議模型 | 掛載位置 | 授權 |
|---|---|---|---|
| 轉錄 | [`Systran/faster-whisper-large-v3-turbo-ct2`](https://huggingface.co/Systran/faster-whisper-large-v3-turbo-ct2) | `WHISPER_MODEL_FOLDER` | MIT |
| 字詞對齊 | WhisperX 依語言選用的 wav2vec2 模型 | `ALIGNMENT_MODEL_FOLDER` | Apache-2.0 / MIT |
| 語者分離 | [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1) | `DIARIZATION_MODEL_FOLDER` | CC-BY-4.0，**gated** |

轉錄模型必須是 CTranslate2 格式。任何轉換過的 Whisper checkpoint 都可以；`large-v3-turbo` 塞不進 GPU 時就換小一點的。

> **語者分離模型是 gated 的。** pyannote pipeline 需要 Hugging Face 帳號並同意模型條款才能下載。`speaker-diarization-community-1` 為 CC-BY-4.0：允許商用，但**必須署名**。請勿把權重 commit 進 repo 或打包進公開 image——那等於繞過 gating，也拿掉了署名。詳見 [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md)。

只有 `ENABLE_DIARIZATION=true` 時才會載入 alignment 與 diarization 模型。關閉時服務照常啟動，並直接拒絕任何要求語者標記的請求。

---

## API

### OpenAI 相容端點

#### `POST /v1/audio/transcriptions`

OpenAI 標準欄位（`multipart/form-data`）：

| 欄位 | 類型 | 預設 | 說明 |
|---|---|---|---|
| `file` | file | 必填 | 待轉錄音檔 |
| `model` | string | – | 僅回填；本服務只載入一份權重，不做選模 |
| `language` | string | 自動 | 未傳或空字串為自動偵測 |
| `prompt` | string | – | 對應 faster-whisper 的 `initial_prompt` |
| `response_format` | string | `json` | `json`、`verbose_json` 或 `text` |
| `timestamp_granularities[]` | string[] | – | 含 `word` 時輸出字級時間戳 |

擴充欄位，由 OpenAI client 經 `extra_body` 傳入：

`beam_size`、`enable_vad`、`use_fast_pipeline`、`condition_on_previous_text`、`repetition_penalty`、`hotwords`、`word_timestamps`、`diarization`、`min_speaker`、`max_speaker`、`always_speaker_label`、`vad_filter_ms`、`vad_threshold`、`neg_threshold`、`min_speech_duration_ms`、`max_speech_duration_s`、`speech_pad_ms`

未傳的欄位沿用環境變數決定的預設值。若請求帶 `X-Request-Id` header，會用作 log 的關聯 ID。

**刻意不支援的欄位。** `temperature` 收下後記 warning 即丟棄：faster-whisper 的預設是一組遞增溫度，只在解碼品質不佳時逐級 fallback；照字面轉發 OpenAI 的預設值 `0` 會關閉整個機制並加重幻覺。`stream=true` 回 400。

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

`verbose_json` 回應含 `task`、`language`、`duration`、`text` 與 `segments[]`。`duration` 取原始音檔長度而非片段時長總和——幻覺過濾會移除片段，用總和會低估長度，而這會讓按時長計價的 proxy 算錯成本。

Speaker label 出現在 `segments[].speaker`（要求字級時間戳時在 `segments[].words[].speaker`），字面為 `SPEAKER_00`、`SPEAKER_01`…… 或 `UNKNOWN`。`text` 永遠不含這些標記。

#### `GET /v1/models`

固定回傳單一 `whisper-1` 別名，因為服務只載入一份權重。存在的目的是讓 OpenAI client 與 LiteLLM 能探測服務。

#### 即時端點

| 端點 | 用途 |
|---|---|
| `POST /v1/audio/chunk_decode` | 單一 WAV 窗的無狀態解碼（multipart） |
| `POST /v1/audio/chunk_decode_final` | 契約相同，另開一條 path 讓 gateway 能單獨分流最後一個 chunk |
| `POST /v1/audio/vad` | 單一 WAV 窗的 speech timestamps（multipart） |

這三個端點與下方原生端點功能相同，但走 `multipart/form-data`，因此能穿過 Envoy、LiteLLM 這類預期檔案上傳的 gateway。

### 原生端點

| 端點 | 用途 |
|---|---|
| `POST /audio/process_audio/` | 批次轉錄，吃伺服器端路徑或下載連結 |
| `POST /audio/chunk_decode/` | 無狀態解碼；request body 為 raw RIFF/WAVE bytes |
| `POST /audio/vad/` | Speech timestamps；request body 為 raw RIFF/WAVE bytes |
| `POST /audio/check-audio-format/` | 轉錄前驗證格式、時長與元資料 |
| `GET /health` | Liveness / readiness probe |
| `GET /metrics` | 固定健康回應，供會輪詢的 dispatcher 使用 |

`/audio/*` 端點以 query string 收參數，錯誤回 `{"detail": ...}`。它們早於 OpenAI 層存在，為既有呼叫端維持穩定。

`POST /audio/vad/` 回傳相對該窗開頭、單位毫秒的語音區間；零個區間表示該窗沒有語音：

```json
{"speech_timestamps": [{"start": 320, "end": 1810}]}
```

### LiteLLM 串接

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

`api_base` 必須以 `/v1` 結尾，LiteLLM 會自行接上 `/audio/transcriptions`。Shuji-Tool 不驗證請求，但 `api_key` 仍必須填——OpenAI SDK 缺 key 會在 client 端就拋錯。若有設 `UVICORN_ROOT_PATH`，`api_base` 要把該前綴寫進去。

---

## 設定

所有設定都是環境變數。可直接複製的完整清單見 [.env.example](.env.example)。

### 模型

| 變數 | 預設 | 說明 |
|---|---|---|
| `WHISPER_MODEL_FOLDER` | `/model/whisper/` | CTranslate2 格式的 Whisper 模型目錄 |
| `ENABLE_DIARIZATION` | `false` | 是否載入 alignment 與 diarization 模型。接受 `true`/`1`/`yes`/`on` 及其否定字詞 |
| `ALIGNMENT_MODEL_FOLDER` | `/model/alignment/` | 需 `ENABLE_DIARIZATION=true` |
| `DIARIZATION_MODEL_FOLDER` | `/model/diarization/` | 需 `ENABLE_DIARIZATION=true` |

### 解碼

| 變數 | 預設 | 說明 |
|---|---|---|
| `WHISPER_BEAM_SIZE` | `5` | 請求未指定時的 beam size |
| `WHISPER_BATCH_SIZE` | `8` | 批次推理管線的 batch size |

### 音檔限制

| 變數 | 預設 | 說明 |
|---|---|---|
| `VTT_SUPPORTED_AUDIO_LENGTHS` | `10800` | 最大時長（秒，預設 3 小時） |
| `SUPPORTED_FORMATS` | `.m4a,.mp3,.wav` | 以逗號分隔的可接受副檔名 |

### 服務

| 變數 | 預設 | 說明 |
|---|---|---|
| `APP_PORT` | `7766` | 監聽埠號 |
| `APP_WORKERS` | `1` | Uvicorn worker 數。每個 worker 各自載入一份模型，建議改用多副本擴展 |
| `UVICORN_ROOT_PATH` | – | 反向代理後的對外路徑前綴 |
| `LOG_LEVEL` | `INFO` | 日誌級別 |

### 幻覺過濾

| 變數 | 預設 | 說明 |
|---|---|---|
| `HARD_NO_SPEECH_PROB_THRESHOLD` | `0.99` | 超過此值，頭尾片段無條件移除 |
| `REMOVE_FIRST_SEGMENT_NO_SPEECH_PROB` | `0.55` | 首片段門檻，與下一項同時成立才移除 |
| `REMOVE_FIRST_SEGMENT_LOG_PROB` | `-1.2` | 首片段 `avg_logprob` 低於或等於此值時移除 |
| `REMOVE_LAST_SEGMENT_DURATION` | `2.5` | 只有短於此秒數的末片段才是候選 |
| `REMOVE_LAST_SEGMENT_NO_SPEECH_PROB` | `0.8` | 末片段門檻 |

砍頭砍尾只作用於批次轉錄。即時 chunk 解碼不做——一個窗的邊界不是一段錄音的邊界——但仍會套用 blacklist。

### 可觀測性

Shuji-Tool 使用標準的 OpenTelemetry 環境變數。除非明確把 `OTEL_SDK_DISABLED` 設為 `false`，telemetry 預設**關閉**。

| 變數 | 預設 | 說明 |
|---|---|---|
| `OTEL_SDK_DISABLED` | `true` | 設為 `false` 才會啟用 traces、metrics、logs |
| `OTEL_SERVICE_NAME` | – | Resource 中的服務名稱 |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | – | 共用的 OTLP endpoint |
| `OTEL_EXPORTER_OTLP_{TRACES,METRICS,LOGS}_ENDPOINT` | – | 單一訊號的覆寫 |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `grpc` | 僅支援 `grpc`；其他值會記 warning 並仍走 gRPC |

每個請求都帶 `flow_id`（`/v1` 取自 `X-Request-Id`，`/audio/*` 由參數明確傳入），並出現在該請求產生的每一行 log，因此單次轉錄可以從頭到尾 grep 出來。

---

## 逐字稿後處理

1. **幻覺 blacklist**：含已知 Whisper 樣板文字（字幕組署名、頻道結語）的片段整段移除。
2. **砍頭砍尾**：頭尾片段的解碼統計顯示為靜音時移除，門檻如上表。
3. **簡體轉繁體**：所有輸出經 OpenCC `s2tw`。轉換以「逐段轉換後串接」實作，保證 `text` 等於 `segments[].text` 的串接。字級輸出必須逐字轉換才能保住時間戳，因此 `words` 串接結果可能與 `text` 略有差異。

第 3 步是無條件執行的。若不需要繁體輸出，請移除 [`src/transcription/service.py`](src/transcription/service.py) 與 [`src/api/openai_schemas.py`](src/api/openai_schemas.py) 中的 OpenCC 轉換。

---

## 架構

```
src/
├── api/
│   ├── app.py                  # FastAPI 入口：路由、telemetry、lifespan
│   ├── openai_router.py        # /v1 端點
│   ├── openai_schemas.py       # verbose_json 組裝、逐段 OpenCC
│   ├── openai_errors.py        # OpenAI 錯誤信封，僅作用於 /v1
│   ├── process_router.py       # /audio/process_audio/
│   ├── chunk_decode_router.py  # /audio/chunk_decode/
│   ├── vad_router.py           # /audio/vad/
│   ├── checker_router.py       # /audio/check-audio-format/
│   ├── checker.py              # 格式、時長、元資料驗證
│   └── dependencies.py         # 引擎注入
├── transcription/
│   ├── engine_loader.py        # 啟動載入與引擎快取
│   ├── engine.py               # FasterWhisperEngine
│   ├── async_faster_whisper.py # asyncio.to_thread 封裝
│   ├── service.py              # 後處理、幻覺過濾、輸出格式化
│   ├── diarization.py          # WhisperX 對齊 + pyannote、Speaker block
│   ├── parameters.py           # WhisperParams / DiarizationParams 與預設值
│   └── cuda_exception.py       # CUDA OOM 分類
├── vad/
│   ├── engine.py               # silero-vad，CPU ONNX，啟動時載入一次
│   └── parameters.py
└── utils/                      # 設定、日誌、音檔、下載、telemetry
```

引擎在 FastAPI lifespan 啟動時建立一次；失敗會讓應用啟動失敗，而不是被吞掉。轉錄透過 `asyncio.to_thread()` 丟進執行緒池，避免阻塞事件迴圈；即時 VAD 引擎跑在 CPU，因此不會和 Whisper 爭 GPU。

決策與被否決的選項記錄在 [`docs/adr/`](docs/adr/)。領域語彙——「Speaker block」、「Speech timestamps」的精確定義以及該避免的說法——在 [`CONTEXT.md`](CONTEXT.md)。

---

## 錯誤代碼

`/v1` 端點回 OpenAI 錯誤信封，錯誤碼保留在 `error.code`：

```json
{"error": {"message": "...", "type": "invalid_request_error", "param": "file", "code": "MSS-VTT-000701"}}
```

`/audio/*` 端點回 `{"detail": {"errorCode": ...}}`。

| 代碼 | 說明 | HTTP |
|---|---|---|
| `MSS-VTT-000000` | 未預期的內部錯誤 | 500 |
| `MSS-VTT-000100` | 時長超限、不支援的格式，或不支援的語言 | 400 / 500 |
| `MSS-VTT-000101` | 缺少必填輸入 | 400 |
| `MSS-VTT-000102` | 不支援的參數值 | 400 |
| `MSS-VTT-000300` | 檔案不存在 | 404 |
| `MSS-VTT-000400` | 下載失敗 | 依下載錯誤而定 |
| `MSS-VTT-000701` | 無效或空的音檔內容 | 400 / 500 |
| `MSS-VTT-000703` | 路徑不存在 | 404 |
| `MSS-VTT-500100` | 未預期的 CUDA 錯誤 | 500 |
| `MSS-VTT-500101` | CUDA 記憶體不足 | 500 |

---

## 開發

```bash
make init     # 安裝依賴
make test     # pytest
make lint     # pylint
make help     # 列出所有 target 與變數
```

測試在 [`tests/conftest.py`](tests/conftest.py) 把所有 GPU 與原生依賴 stub 掉，因此不需要 CUDA、PyTorch 或模型權重也能在任何機器上跑。`requirements-test.txt` 就是這組輕量依賴，CI 安裝的也是它。

### Docker build stage

| Stage | 內容 | 建置指令 |
|---|---|---|
| `runtime` | 純 Python、非 root——預設 stage | `make image` |
| `dev` | 含建置工具與測試依賴 | `STAGE=dev make image` |
| `final` | Nuitka 編譯的二進位、非 root | `STAGE=final make image` |
| `base` | 共用的依賴層 | – |

Image 座標可用 `IMAGE_REGISTRY`、`IMAGE_NAME`、`IMAGE_VERSION` 覆寫；`make push` 會先建置再推送。Image 的 `TZ` 預設為 `UTC`，若希望 log 時間戳為本地時間，建置時加 `--build-arg TZ=Asia/Taipei`。

---

## 安全性

本開源版不提供即時弱點修復；安全修補依本專案公開的支援版本與發布週期處理。

## 授權

Apache License 2.0——見 [LICENSE](LICENSE) 與 [NOTICE](NOTICE)。

第三方元件與其授權列於 [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md)。Python 套件皆為寬鬆授權；GPU 推論所需的 NVIDIA CUDA 函式庫屬 NVIDIA 專有授權，由你安裝時自行下載，本專案不散布。模型權重由各自作者獨立授權、不隨本專案散布——其中語者分離 pipeline 是 gated 且要求署名。
