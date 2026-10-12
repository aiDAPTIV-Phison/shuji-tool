"""OpenAI 相容的轉錄端點。

提供 `POST /v1/audio/transcriptions`、`GET /v1/models`，以及即時用的 multipart
`POST /v1/audio/chunk_decode` 與 `POST /v1/audio/vad`，讓 LiteLLM 以 openai
provider 直接串接。除 OpenAI 標準欄位外另收 faster-whisper 專屬參數，呼叫端可
透過 openai-python 的 `extra_body` 帶入（LiteLLM 會把非白名單參數收進
extra_body，再由 openai-python 攤平回 multipart form 欄位）。
"""

import shutil
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from api.dependencies import get_openai_transcription_service, get_vad_engine
from api.openai_errors import OpenAICompatError
from api.openai_schemas import (
    RESPONSE_FORMAT_TEXT,
    RESPONSE_FORMAT_VERBOSE_JSON,
    SUPPORTED_RESPONSE_FORMATS,
    TranscriptionJsonResponse,
    build_text,
    build_verbose_response,
)
from transcription.cuda_exception import CudaException, CudaOutOfMemoryError
from transcription.parameters import DiarizationParams, get_default_whisper_params, normalize_language
from transcription.service import TranscriptionService
from utils.LogHandler import LogHandler as lh
from utils.call_site import get_call_site_info
from utils.logger import Logger
from vad.engine import SileroVadEngine
from vad.parameters import VadParams

router = APIRouter()
logger = Logger(name="OpenAICompatLogger", log_file="logs/openai_compat.log")

DEFAULT_MODEL_ID = "whisper-1"
DEFAULT_SPEAKER_LABEL = "SPEAKER_00"
UPLOAD_CHUNK_SIZE = 1 << 20
_SERVICE_START_TS = int(time.time())


def _resolve_flow_id(request_id: Optional[str]) -> str:
    """決定本次請求的 flow_id。

    OpenAI 契約沒有 flow_id，因此優先沿用上游的 X-Request-Id 以串接 trace，
    缺少時自行產生。

    Args:
        request_id: X-Request-Id 標頭值。

    Returns:
        flow_id 字串。
    """
    if request_id and request_id.strip():
        return request_id.strip()
    return uuid4().hex


def _resolve_response_format(response_format: Optional[str]) -> str:
    """驗證並回傳 response_format。

    Args:
        response_format: 呼叫端指定的格式；未傳時視為 json。

    Returns:
        已驗證的格式字串。

    Raises:
        OpenAICompatError: 格式不支援時回 400。
    """
    resolved = (response_format or "json").strip().lower()
    if resolved not in SUPPORTED_RESPONSE_FORMATS:
        raise OpenAICompatError(
            status_code=400,
            message=(
                f"response_format '{resolved}' is not supported. "
                f"Supported formats: {', '.join(SUPPORTED_RESPONSE_FORMATS)}."
            ),
            code="MSS-VTT-000102",
            param="response_format",
        )
    return resolved


def _wants_word_timestamps(
    granularities: Optional[list[str]],
    word_timestamps: Optional[bool],
) -> bool:
    """決定是否輸出字級時間戳。

    明確傳入的 word_timestamps 優先於 OpenAI 的 timestamp_granularities。

    Args:
        granularities: timestamp_granularities 清單。
        word_timestamps: faster-whisper 專屬的 word_timestamps 覆寫值。

    Returns:
        需要字級時間戳時為 True。
    """
    if word_timestamps is not None:
        return word_timestamps
    return any(item.strip().lower() == "word" for item in granularities or [])


async def _spool_upload(file: UploadFile, directory: Path) -> Path:
    """分塊寫入上傳音檔，避免大檔一次讀進記憶體。

    保留原始副檔名讓解碼器有正確提示；無副檔名時預設 .wav。

    Args:
        file: 上傳的音檔。
        directory: 落檔目錄。

    Returns:
        落檔後的路徑。

    Raises:
        OpenAICompatError: 檔案內容為空時回 400。
    """
    suffix = Path(file.filename or "").suffix or ".wav"
    audio_path = directory / f"{uuid4().hex}{suffix}"
    written = 0
    with open(audio_path, "wb") as target:
        while chunk := await file.read(UPLOAD_CHUNK_SIZE):
            target.write(chunk)
            written += len(chunk)

    if written == 0:
        raise OpenAICompatError(
            status_code=400,
            message="Uploaded audio file is empty.",
            code="MSS-VTT-000701",
            param="file",
        )
    return audio_path


def _wants_speaker_labels(
    diarization: Optional[bool], always_speaker_label: Optional[bool]
) -> bool:
    """判斷請求是否要求輸出 Speaker label。

    Args:
        diarization: 是否開啟 Diarization。
        always_speaker_label: 是否強制單一 Speaker label。

    Returns:
        任一語者旗標為 True 時為 True。
    """
    return bool(diarization) or bool(always_speaker_label)


def _stamp_always_speaker_label(segments: list[Any]) -> list[Any]:
    """在 faster-whisper 片段與字級時間戳上蓋 SPEAKER_00，不合併。

    Args:
        segments: 轉錄片段。

    Returns:
        帶 speaker 屬性的 duck-typed 片段清單。
    """
    stamped = []
    for segment in segments:
        words = None
        raw_words = getattr(segment, "words", None)
        if raw_words:
            words = [
                SimpleNamespace(
                    word=word.word,
                    start=word.start,
                    end=word.end,
                    speaker=DEFAULT_SPEAKER_LABEL,
                )
                for word in raw_words
            ]
        stamped.append(
            SimpleNamespace(
                id=getattr(segment, "id", None),
                seek=getattr(segment, "seek", 0),
                start=segment.start,
                end=segment.end,
                text=segment.text,
                tokens=getattr(segment, "tokens", None),
                temperature=getattr(segment, "temperature", 0.0),
                avg_logprob=getattr(segment, "avg_logprob", 0.0),
                compression_ratio=getattr(segment, "compression_ratio", 0.0),
                no_speech_prob=getattr(segment, "no_speech_prob", 0.0),
                words=words,
                speaker=DEFAULT_SPEAKER_LABEL,
            )
        )
    return stamped


def _whisperx_word_to_namespace(word: dict[str, Any], fallback_speaker: str) -> SimpleNamespace:
    """將 WhisperX 字級 dict 轉成 duck-typed 物件。

    Args:
        word: WhisperX word dict。
        fallback_speaker: 字上缺少 speaker 時的 Speaker label。

    Returns:
        帶 word／start／end／speaker 的 SimpleNamespace。
    """
    return SimpleNamespace(
        word=word["word"],
        start=word["start"],
        end=word["end"],
        speaker=word.get("speaker") or fallback_speaker,
    )


def _whisperx_segments_to_namespaces(
    wx_segments: list[dict[str, Any]],
    include_words: bool,
    merge_blocks,
) -> list[Any]:
    """將 WhisperX 片段轉成 build_verbose_response 可吃的 duck-typed 片段。

    未要求字級時間戳時先合併為 Speaker block；要求時保留對齊原片段。

    Args:
        wx_segments: WhisperX 標記後的片段。
        include_words: 是否輸出字級時間戳。
        merge_blocks: 將連續同語者片段合併為 Speaker block 的函式。

    Returns:
        duck-typed 片段清單。
    """
    if not include_words:
        blocks = merge_blocks(wx_segments)
        return [
            SimpleNamespace(
                start=block["start"],
                end=block["end"],
                text=block["text"],
                speaker=block["speaker"],
                words=None,
            )
            for block in blocks
        ]

    result = []
    for segment in wx_segments:
        text = (segment.get("text") or "").strip()
        if not text:
            continue
        speaker = segment.get("speaker") or "UNKNOWN"
        words = None
        raw_words = segment.get("words") or []
        if raw_words:
            words = [
                _whisperx_word_to_namespace(word, speaker)
                for word in raw_words
                if word.get("start") is not None and word.get("end") is not None
            ]
        result.append(
            SimpleNamespace(
                start=segment.get("start", 0.0),
                end=segment.get("end", 0.0),
                text=text,
                speaker=speaker,
                words=words,
            )
        )
    return result


def _http_exception_error_code(exc: HTTPException) -> Optional[str]:
    """從 HTTPException.detail 取出 MSS-VTT 錯誤碼。

    Args:
        exc: FastAPI HTTPException。

    Returns:
        errorCode 字串；detail 非 dict 或缺少時為 None。
    """
    detail = exc.detail
    if isinstance(detail, dict):
        return detail.get("errorCode")
    return None


def _require_wav_container(audio_path: Path) -> None:
    """確認落檔內容為 RIFF/WAVE，否則以 OpenAI 信封回 400。

    Args:
        audio_path: 已 spool 的音檔路徑。

    Raises:
        OpenAICompatError: 檔頭非 RIFF/WAVE 時回 400。
    """
    with open(audio_path, "rb") as handle:
        header = handle.read(12)
    if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
        raise OpenAICompatError(
            status_code=400,
            message="Invalid or empty WAV body.",
            code="MSS-VTT-000701",
            param="file",
        )


@router.get("/models")
async def list_models():
    """列出可用模型，供 LiteLLM 與 OpenAI client 探測。

    本服務只載入一份權重，因此固定回傳單一 whisper-1 別名。

    Returns:
        OpenAI 模型清單格式的 JSON。
    """
    return {
        "object": "list",
        "data": [
            {
                "id": DEFAULT_MODEL_ID,
                "object": "model",
                "created": _SERVICE_START_TS,
                "owned_by": "shuji-tool",
            }
        ],
    }


@router.post("/audio/transcriptions", response_model=None)
async def create_transcription(
    file: UploadFile = File(..., description="待轉錄的音檔"),
    model: Optional[str] = Form(None, description="模型名稱；本服務僅回填不做選模"),
    language: Optional[str] = Form(None, description="轉錄語言代碼；未傳或空字串為自動偵測"),
    prompt: Optional[str] = Form(None, description="對應 faster-whisper 的 initial_prompt"),
    response_format: Optional[str] = Form(None, description="json、verbose_json 或 text"),
    temperature: Optional[float] = Form(None, description="不支援；收下後忽略"),
    timestamp_granularities: Optional[list[str]] = Form(
        None, alias="timestamp_granularities[]", description="含 word 時輸出字級時間戳"
    ),
    timestamp_granularities_plain: Optional[list[str]] = Form(
        None, alias="timestamp_granularities", description="未帶方括號的欄位名相容寫法"
    ),
    stream: Optional[bool] = Form(None, description="不支援；傳 true 時回 400"),
    # 以下為 faster-whisper 專屬參數，透過 extra_body 傳入
    beam_size: Optional[int] = Form(None, description="解碼 beam size"),
    enable_vad: Optional[bool] = Form(None, description="是否啟用 Whisper 內建 VAD 過濾"),
    use_fast_pipeline: Optional[bool] = Form(None, description="是否使用批次加速管線"),
    condition_on_previous_text: Optional[bool] = Form(None, description="segment 是否互為上文"),
    repetition_penalty: Optional[float] = Form(None, description="重複懲罰"),
    hotwords: Optional[str] = Form(None, description="熱詞"),
    word_timestamps: Optional[bool] = Form(None, description="字級時間戳；優先於 timestamp_granularities"),
    always_speaker_label: Optional[bool] = Form(None, description="是否強制加上 SPEAKER_00 標籤"),
    diarization: Optional[bool] = Form(None, description="是否啟用語者分離"),
    min_speaker: Optional[int] = Form(None, description="最小說話者數量；僅 diarization=true 時生效"),
    max_speaker: Optional[int] = Form(None, description="最大說話者數量；僅 diarization=true 時生效"),
    vad_filter_ms: Optional[int] = Form(None, description="VAD 最小靜音時長"),
    vad_threshold: Optional[float] = Form(None, description="VAD 閾值"),
    neg_threshold: Optional[float] = Form(None, description="VAD 負閾值"),
    min_speech_duration_ms: Optional[int] = Form(None, description="VAD 最小語音時長"),
    max_speech_duration_s: Optional[float] = Form(None, description="VAD 最大語音時長"),
    speech_pad_ms: Optional[int] = Form(None, description="VAD 語音填充時長"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
    transcription_service: TranscriptionService = Depends(get_openai_transcription_service),
):
    """以 OpenAI 契約轉錄單一音檔。

    Args:
        file: 待轉錄的音檔。
        model: 模型名稱；僅回填，未知名稱會記 warning 後沿用預設參數。
        language: 轉錄語言代碼；未傳或空字串為自動偵測。
        prompt: 對應 initial_prompt。
        response_format: json、verbose_json 或 text；未傳為 json。
        temperature: 不支援，收下後忽略並記 warning。
        timestamp_granularities: openai-python 送出的帶方括號欄位。
        timestamp_granularities_plain: 未帶方括號的相容寫法。
        stream: 不支援，傳 true 時回 400。
        beam_size: 解碼 beam size。
        enable_vad: 是否啟用 Whisper 內建 VAD 過濾。
        use_fast_pipeline: 是否使用批次加速管線。
        condition_on_previous_text: segment 是否互為上文。
        repetition_penalty: 重複懲罰。
        hotwords: 熱詞。
        word_timestamps: 字級時間戳開關，優先於 timestamp_granularities。
        always_speaker_label: 是否強制加上 SPEAKER_00 標籤。
        diarization: 是否啟用語者分離。
        min_speaker: 最小說話者數量。
        max_speaker: 最大說話者數量。
        vad_filter_ms: VAD 最小靜音時長。
        vad_threshold: VAD 閾值。
        neg_threshold: VAD 負閾值。
        min_speech_duration_ms: VAD 最小語音時長。
        max_speech_duration_s: VAD 最大語音時長。
        speech_pad_ms: VAD 語音填充時長。
        x_request_id: 上游 trace 識別；缺少時自行產生 flow_id。
        transcription_service: 轉錄服務。

    Returns:
        依 response_format 回 json、verbose_json 或純文字。
    """
    callSite = get_call_site_info()
    flow_id = _resolve_flow_id(x_request_id)
    resolved_format = _resolve_response_format(response_format)

    if stream:
        raise OpenAICompatError(
            status_code=400,
            message="Streaming transcription is not supported by this model.",
            code="MSS-VTT-000102",
            param="stream",
        )

    wants_speakers = _wants_speaker_labels(diarization, always_speaker_label)
    if wants_speakers and resolved_format != RESPONSE_FORMAT_VERBOSE_JSON:
        raise OpenAICompatError(
            status_code=400,
            message=(
                "Speaker labels require response_format=verbose_json. "
                f"Got '{resolved_format}'."
            ),
            code="MSS-VTT-000102",
            param="response_format",
        )

    if diarization:
        diarization_engine = transcription_service.diarization_engine
        if diarization_engine is None or not diarization_engine.is_enable:
            raise OpenAICompatError(
                status_code=400,
                message="Diarization is not available on this deployment.",
                code="MSS-VTT-000102",
                param="diarization",
            )

    if temperature is not None:
        lh.handle_warning(
            logger,
            flow_id,
            callSite,
            f"temperature is not supported and will be ignored. flow_id={flow_id}, temperature={temperature}",
        )

    if model and model != DEFAULT_MODEL_ID:
        lh.handle_warning(
            logger,
            flow_id,
            callSite,
            f"Unknown model requested; falling back to default decode settings. "
            f"flow_id={flow_id}, model={model}",
        )

    granularities = timestamp_granularities or timestamp_granularities_plain
    include_words = _wants_word_timestamps(granularities, word_timestamps)

    param_updates = {
        "language": normalize_language(language),
        "initial_prompt": prompt,
        "word_timestamps": include_words,
    }
    optional_updates = {
        "beam_size": beam_size,
        "enable_vad": enable_vad,
        "use_fast_pipeline": use_fast_pipeline,
        "condition_on_previous_text": condition_on_previous_text,
        "repetition_penalty": repetition_penalty,
        "hotwords": hotwords,
        "always_speaker_label": always_speaker_label,
        "vad_filter_ms": vad_filter_ms,
        "vad_threshold": vad_threshold,
        "neg_threshold": neg_threshold,
        "min_speech_duration_ms": min_speech_duration_ms,
        "max_speech_duration_s": max_speech_duration_s,
        "speech_pad_ms": speech_pad_ms,
    }
    param_updates.update({k: v for k, v in optional_updates.items() if v is not None})
    params = get_default_whisper_params().model_copy(update=param_updates)

    lh.handle_info(
        logger,
        flow_id,
        callSite,
        (
            f"Received OpenAI transcription request. flow_id={flow_id}, model={model}, "
            f"filename={file.filename}, response_format={resolved_format}, "
            f"language={params.language}, word_timestamps={include_words}, "
            f"diarization={bool(diarization)}, always_speaker_label={bool(always_speaker_label)}"
        ),
    )

    temp_dir = Path(tempfile.mkdtemp(prefix="openai_transcribe_"))
    try:
        audio_path = await _spool_upload(file, temp_dir)
        segments, info = await transcription_service.transcribe_to_segments(
            flow_id=flow_id, audio_path=str(audio_path), params=params
        )

        if diarization:
            diarization_params = DiarizationParams(
                diarization=True,
                min_speaker=min_speaker,
                max_speaker=max_speaker,
            )
            wx_segments = transcription_service.diarization_engine.diarize_to_segments(
                flow_id,
                segments,
                info,
                str(audio_path),
                diarization_params,
            )
            segments = _whisperx_segments_to_namespaces(
                wx_segments,
                include_words,
                transcription_service.diarization_engine.merge_speaker_blocks,
            )
        elif always_speaker_label:
            segments = _stamp_always_speaker_label(segments)

        if resolved_format == RESPONSE_FORMAT_VERBOSE_JSON:
            payload = build_verbose_response(segments, info, include_words)
            return JSONResponse(content=payload.model_dump(exclude_none=True))

        text = build_text(segments)
        if resolved_format == RESPONSE_FORMAT_TEXT:
            return PlainTextResponse(content=text)
        return JSONResponse(content=TranscriptionJsonResponse(text=text).model_dump())

    except OpenAICompatError:
        raise
    except HTTPException as e:
        error_code = _http_exception_error_code(e)
        if error_code == "MSS-VTT-000100":
            message = e.detail.get("message", str(e.detail)) if isinstance(e.detail, dict) else str(e.detail)
            raise OpenAICompatError(
                status_code=400,
                message=message,
                code="MSS-VTT-000100",
                param="language",
            ) from e
        raise OpenAICompatError(
            status_code=e.status_code,
            message=str(e.detail),
            code=error_code or "MSS-VTT-000000",
        ) from e
    except CudaOutOfMemoryError as e:
        logger.error(f"CUDA out of memory. flow_id={flow_id}, error={str(e)}")
        raise OpenAICompatError(
            status_code=500,
            message="Transcription failed: GPU out of memory.",
            code="MSS-VTT-500101",
        ) from e
    except CudaException as e:
        logger.error(f"Unexpected CUDA error. flow_id={flow_id}, error={str(e)}")
        raise OpenAICompatError(
            status_code=500,
            message="Transcription failed: unexpected CUDA error.",
            code="MSS-VTT-500100",
        ) from e
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during OpenAI transcription. "
            f"flow_id={flow_id}, error={str(e)}"
        )
        raise OpenAICompatError(
            status_code=500,
            message=f"Transcription failed: {str(e)}",
            code="MSS-VTT-000000",
        ) from e
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

@router.post("/audio/chunk_decode_final", response_model=None)
async def create_chunk_decode_final(
    file: UploadFile = File(..., description="待解碼的 WAV chunk"),
    flow_id: str = Form(..., description="RealtimeSession 生命週期識別"),
    sequence_no: int = Form(..., ge=0, description="同一 flow_id 內由 0 遞增的序號"),
    language: Optional[str] = Form(None, description="轉錄語言代碼；未傳或空字串為自動偵測"),
    initial_prompt: Optional[str] = Form(None, description="延續上下文，語意比照批次 initial_prompt"),
    condition_on_previous_text: bool = Form(True, description="同一次解碼內 segment 是否互為上文"),
    is_final: Optional[bool] = Form(None, description="partial／final 標示；僅供 log，不改解碼"),
    beam_size: Optional[int] = Form(None, description="解碼 beam size；未傳則用 process 預設"),
    use_fast_pipeline: Optional[bool] = Form(None, description="是否使用 fast pipeline；未傳則用 process 預設"),
    repetition_penalty: Optional[float] = Form(None, description="重複懲罰；未傳則用 process 預設"),
    hotwords: Optional[str] = Form(None, description="熱詞參數"),
    diarization: Optional[bool] = Form(None, description="為 true 時對本包音做 WhisperX diarization"),
    transcription_service: TranscriptionService = Depends(get_openai_transcription_service),
):
    """用於辨別最後一個 chunk 是否結束，並回傳最終逐字稿。此端點用於第三方分流服務(Envoy Gateway, LiteLLM)"""
    return await create_chunk_decode(file, flow_id, sequence_no, language, initial_prompt, condition_on_previous_text, is_final, beam_size, use_fast_pipeline, repetition_penalty, hotwords, diarization, transcription_service)

@router.post("/audio/chunk_decode", response_model=None)
async def create_chunk_decode(
    file: UploadFile = File(..., description="待解碼的 WAV chunk"),
    flow_id: str = Form(..., description="RealtimeSession 生命週期識別"),
    sequence_no: int = Form(..., ge=0, description="同一 flow_id 內由 0 遞增的序號"),
    language: Optional[str] = Form(None, description="轉錄語言代碼；未傳或空字串為自動偵測"),
    initial_prompt: Optional[str] = Form(None, description="延續上下文，語意比照批次 initial_prompt"),
    condition_on_previous_text: bool = Form(True, description="同一次解碼內 segment 是否互為上文"),
    is_final: Optional[bool] = Form(None, description="partial／final 標示；僅供 log，不改解碼"),
    beam_size: Optional[int] = Form(None, description="解碼 beam size；未傳則用 process 預設"),
    use_fast_pipeline: Optional[bool] = Form(None, description="是否使用 fast pipeline；未傳則用 process 預設"),
    repetition_penalty: Optional[float] = Form(None, description="重複懲罰；未傳則用 process 預設"),
    hotwords: Optional[str] = Form(None, description="熱詞參數"),
    diarization: Optional[bool] = Form(None, description="為 true 時對本包音做 WhisperX diarization"),
    transcription_service: TranscriptionService = Depends(get_openai_transcription_service),
):
    """以 multipart 接收單一 wav chunk，做無狀態解碼並回傳文字。

    Args:
        file: 待解碼的 WAV chunk。
        flow_id: RealtimeSession 生命週期識別。
        sequence_no: 同一 flow_id 內由 0 遞增的序號。
        language: 轉錄語言代碼；未傳或空字串為自動偵測。
        initial_prompt: 前次逐字稿延續上下文。
        condition_on_previous_text: 同一次解碼內 segment 是否互為上文。
        is_final: partial/final 標示；僅寫入 log，不改變解碼演算法。
        beam_size: 解碼 beam size；未傳則用 WHISPER_BEAM_SIZE。
        use_fast_pipeline: 是否使用 fast pipeline；未傳則為 False。
        repetition_penalty: 重複懲罰；未傳則為 1.0。
        hotwords: 熱詞；未傳則為 None。
        transcription_service: 轉錄服務。

    Returns:
        僅含 text 的 JSON。
    """
    callSite = get_call_site_info()
    language = normalize_language(language)
    lh.handle_debug(
        logger,
        flow_id,
        callSite,
        (
            f"Received chunk decode request. flow_id={flow_id}, "
            f"sequence_no={sequence_no}, language={language}, is_final={is_final}"
        ),
    )

    param_updates = {
        "language": language,
        "initial_prompt": initial_prompt,
        "condition_on_previous_text": condition_on_previous_text,
        "enable_vad": False,
    }
    if beam_size is not None:
        param_updates["beam_size"] = beam_size
    if use_fast_pipeline is not None:
        param_updates["use_fast_pipeline"] = use_fast_pipeline
    if repetition_penalty is not None:
        param_updates["repetition_penalty"] = repetition_penalty
    if hotwords is not None:
        param_updates["hotwords"] = hotwords
    params = get_default_whisper_params().model_copy(update=param_updates)

    temp_dir = Path(tempfile.mkdtemp(prefix="openai_chunk_decode_"))
    try:
        audio_path = await _spool_upload(file, temp_dir)
        _require_wav_container(audio_path)
        result = await transcription_service.decode_chunk(
            flow_id=flow_id,
            audio_path=str(audio_path),
            params=params,
            diarization=bool(diarization),
        )
        result_text = result.transcription
        lh.handle_info( logger, flow_id, callSite, f"Chunk decode completed. flow_id={flow_id}, result_text={result_text}" )
        # 維持 OpenAI json 的 text；turns／speakers 僅在兩者都有時附加
        content = TranscriptionJsonResponse(text=result_text).model_dump()
        if result.turns is not None and result.speakers is not None:
            content["turns"] = result.turns
            content["speakers"] = result.speakers
        return JSONResponse(content=content)

    except OpenAICompatError:
        raise
    except HTTPException as e:
        error_code = _http_exception_error_code(e)
        if error_code == "MSS-VTT-000100":
            message = e.detail.get("message", str(e.detail)) if isinstance(e.detail, dict) else str(e.detail)
            raise OpenAICompatError(
                status_code=400,
                message=message,
                code="MSS-VTT-000100",
                param="language",
            ) from e
        raise OpenAICompatError(
            status_code=e.status_code,
            message=str(e.detail),
            code=error_code or "MSS-VTT-000000",
        ) from e
    except CudaOutOfMemoryError as e:
        logger.error(f"CUDA out of memory. flow_id={flow_id}, error={str(e)}")
        raise OpenAICompatError(
            status_code=500,
            message="Chunk decode failed: GPU out of memory.",
            code="MSS-VTT-500101",
        ) from e
    except CudaException as e:
        logger.error(f"Unexpected CUDA error. flow_id={flow_id}, error={str(e)}")
        raise OpenAICompatError(
            status_code=500,
            message="Chunk decode failed: unexpected CUDA error.",
            code="MSS-VTT-500100",
        ) from e
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during chunk decode. "
            f"flow_id={flow_id}, error={str(e)}"
        )
        raise OpenAICompatError(
            status_code=500,
            message=f"Chunk decode failed: {str(e)}",
            code="MSS-VTT-000000",
        ) from e
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@router.post("/audio/vad", response_model=None)
async def create_vad(
    file: UploadFile = File(..., description="待偵測的 WAV chunk"),
    flow_id: str = Form(..., description="RealtimeSession 生命週期識別"),
    sequence_no: Optional[int] = Form(None, ge=0, description="同一 flow_id 內序號；選填，僅供 log"),
    vad_threshold: Optional[float] = Form(None, description="語音活動檢測閾值"),
    vad_filter_ms: Optional[int] = Form(None, description="最小靜音時長（對應 min_silence_duration_ms）"),
    neg_threshold: Optional[float] = Form(None, description="語音活動檢測負閾值"),
    min_speech_duration_ms: Optional[int] = Form(None, description="最小語音時長"),
    max_speech_duration_s: Optional[float] = Form(None, description="最大語音時長"),
    speech_pad_ms: Optional[int] = Form(None, description="語音填充時長"),
    vad_engine: SileroVadEngine = Depends(get_vad_engine),
):
    """以 multipart 接收單一 wav chunk，回傳相對該窗開頭的 Speech timestamps。

    Args:
        file: 待偵測的 WAV chunk。
        flow_id: RealtimeSession 生命週期識別。
        sequence_no: 同一 flow_id 內序號；選填，僅供 log。
        vad_threshold: 選填 Silero threshold。
        vad_filter_ms: 選填 min_silence_duration_ms。
        neg_threshold: 選填 neg_threshold。
        min_speech_duration_ms: 選填最小語音時長。
        max_speech_duration_s: 選填最大語音時長。
        speech_pad_ms: 選填語音填充。
        vad_engine: Silero VAD 引擎。

    Returns:
        僅含 speech_timestamps 的 JSON。
    """
    callSite = get_call_site_info()
    lh.handle_debug(
        logger,
        flow_id,
        callSite,
        (
            f"Received vad request. flow_id={flow_id}, "
            f"sequence_no={sequence_no}"
        ),
    )

    params = VadParams(
        vad_threshold=vad_threshold,
        vad_filter_ms=vad_filter_ms,
        neg_threshold=neg_threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        max_speech_duration_s=max_speech_duration_s,
        speech_pad_ms=speech_pad_ms,
    )

    temp_dir = Path(tempfile.mkdtemp(prefix="openai_vad_"))
    try:
        audio_path = await _spool_upload(file, temp_dir)
        _require_wav_container(audio_path)
        speech_timestamps = await vad_engine.detect_speech_timestamps(
            flow_id=flow_id, audio_path=str(audio_path), params=params
        )
        return JSONResponse(content={"speech_timestamps": speech_timestamps})

    except OpenAICompatError:
        raise
    except HTTPException as e:
        error_code = _http_exception_error_code(e)
        raise OpenAICompatError(
            status_code=e.status_code,
            message=str(e.detail),
            code=error_code or "MSS-VTT-000000",
        ) from e
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during vad. "
            f"flow_id={flow_id}, error={str(e)}"
        )
        raise OpenAICompatError(
            status_code=500,
            message=f"VAD failed: {str(e)}",
            code="MSS-VTT-000000",
        ) from e
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
