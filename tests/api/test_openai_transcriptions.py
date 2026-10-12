import io
import wave
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from api.dependencies import get_openai_transcription_service
from api.openai_errors import (
    OpenAICompatError,
    openai_error_handler,
    openai_validation_error_handler,
)
from api.openai_router import router
from transcription.service import TranscriptionService

ENDPOINT = "/v1/audio/transcriptions"
MODELS_ENDPOINT = "/v1/models"


def _minimal_wav() -> bytes:
    """產生最短合法 RIFF/WAVE bytes，供 multipart 上傳使用。

    Returns:
        16 kHz mono PCM wav。
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 160)
    return buf.getvalue()


def _word(word: str, start: float, end: float):
    """組出測試用字級時間戳。

    Args:
        word: 字詞文字。
        start: 起始秒。
        end: 結束秒。

    Returns:
        帶 word/start/end 的 SimpleNamespace。
    """
    return SimpleNamespace(word=word, start=start, end=end)


def _segment(
    text: str,
    *,
    seg_id: int = 0,
    seek: int = 0,
    start: float = 0.0,
    end: float = 3.0,
    tokens=None,
    temperature: float = 0.0,
    avg_logprob: float = -0.1,
    compression_ratio: float = 1.2,
    no_speech_prob: float = 0.01,
    words=None,
):
    """組出測試用 segment（duck typing，不載入 faster-whisper）。

    預設的 no_speech_prob 與 avg_logprob 刻意遠離砍頭砍尾門檻，讓片段保留。

    Args:
        text: 片段文字。
        seg_id: 片段 id。
        seek: 解碼起點。
        start: 起始秒。
        end: 結束秒。
        tokens: token id 清單。
        temperature: 解碼溫度。
        avg_logprob: 平均 log 機率。
        compression_ratio: 壓縮比。
        no_speech_prob: 無語音機率。
        words: 字級時間戳清單。

    Returns:
        帶完整 verbose_json 欄位的 SimpleNamespace。
    """
    return SimpleNamespace(
        id=seg_id,
        seek=seek,
        start=start,
        end=end,
        text=text,
        tokens=tokens if tokens is not None else [100, 200],
        temperature=temperature,
        avg_logprob=avg_logprob,
        compression_ratio=compression_ratio,
        no_speech_prob=no_speech_prob,
        words=words,
    )


def _info(language: str = "zh", duration: float = 12.5):
    """組出測試用轉錄資訊。

    Args:
        language: 偵測語言。
        duration: 原始音檔長度。

    Returns:
        帶 language/language_probability/duration 的 SimpleNamespace。
    """
    return SimpleNamespace(language=language, language_probability=0.98, duration=duration)


class FakeEngine:
    """回傳固定 segment 的引擎。"""

    def __init__(self, segments, info=None):
        """保存預設解碼結果。

        Args:
            segments: 回傳的片段清單。
            info: 轉錄資訊；省略則用預設值。
        """
        self._segments = segments
        self._info = info if info is not None else _info()

    async def transcribe_async_method(self, flow_id, audio_path, params):
        """回傳建構時給定的片段。

        Args:
            flow_id: 流程識別，測試忽略。
            audio_path: 落檔路徑，測試忽略。
            params: 轉錄參數，測試忽略。

        Returns:
            (segments, info)。
        """
        del flow_id, audio_path, params
        return self._segments, self._info


class EchoParamsEngine:
    """把指定的 params 欄位回寫成片段文字，用來觀察參數是否傳到引擎。"""

    def __init__(self, formatter):
        """保存文字組裝函式。

        Args:
            formatter: 接受 (flow_id, params) 並回傳字串。
        """
        self._formatter = formatter

    async def transcribe_async_method(self, flow_id, audio_path, params):
        """將 formatter 結果作為唯一片段文字回傳。

        Args:
            flow_id: 流程識別。
            audio_path: 落檔路徑，測試忽略。
            params: 路由組出的轉錄參數。

        Returns:
            (segments, info)。
        """
        del audio_path
        return [_segment(self._formatter(flow_id, params))], _info()


def _client_with_engine(engine, diarization_engine=None) -> TestClient:
    """組出只掛 OpenAI 相容路由並注入指定引擎的 TestClient。

    Args:
        engine: 轉錄引擎實例。
        diarization_engine: 可選的語者分離引擎。

    Returns:
        FastAPI TestClient。
    """
    app = FastAPI()
    app.include_router(router, prefix="/v1")
    app.add_exception_handler(OpenAICompatError, openai_error_handler)
    app.add_exception_handler(RequestValidationError, openai_validation_error_handler)
    service = TranscriptionService(engine=engine, diarization_engine=diarization_engine)
    app.dependency_overrides[get_openai_transcription_service] = lambda: service
    return TestClient(app)


def _client(segments=None, info=None) -> TestClient:
    """組出注入 FakeEngine 的 TestClient。

    Args:
        segments: FakeEngine 回傳的片段；省略則為單段 `hello`。
        info: 轉錄資訊。

    Returns:
        FastAPI TestClient。
    """
    return _client_with_engine(FakeEngine(segments or [_segment("hello")], info))


def _upload(**data):
    """組出 multipart 上傳參數。

    Args:
        **data: 額外的 form 欄位。

    Returns:
        傳給 TestClient.post 的 kwargs。
    """
    return {
        "files": {"file": ("sample.wav", _minimal_wav(), "audio/wav")},
        "data": data,
    }


def test_list_models_returns_whisper_1():
    """/v1/models 回傳單一 whisper-1 別名。"""
    client = _client()
    response = client.get(MODELS_ENDPOINT)
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert [item["id"] for item in body["data"]] == ["whisper-1"]


def test_default_format_returns_text_only():
    """未傳 response_format 時回 json，僅含 text 欄位。"""
    client = _client([_segment("hello")])
    response = client.post(ENDPOINT, **_upload(model="whisper-1"))
    assert response.status_code == 200
    assert response.json() == {"text": "hello"}


def test_text_format_returns_plain_text():
    """response_format=text 回純文字而非 JSON。"""
    client = _client([_segment("hello")])
    response = client.post(ENDPOINT, **_upload(model="whisper-1", response_format="text"))
    assert response.status_code == 200
    assert response.text == "hello"
    assert response.headers["content-type"].startswith("text/plain")


def test_verbose_json_returns_openai_shape():
    """verbose_json 回應含 OpenAI 契約的所有欄位，segment 欄位對齊 faster-whisper。"""
    client = _client(
        [_segment("hello", seg_id=0, seek=0, start=0.5, end=2.5, tokens=[1, 2, 3])],
        info=_info(language="en", duration=9.0),
    )
    response = client.post(ENDPOINT, **_upload(model="whisper-1", response_format="verbose_json"))
    assert response.status_code == 200
    body = response.json()
    assert body["task"] == "transcribe"
    assert body["language"] == "en"
    assert body["duration"] == 9.0
    assert body["text"] == "hello"
    assert len(body["segments"]) == 1
    segment = body["segments"][0]
    assert segment == {
        "id": 0,
        "seek": 0,
        "start": 0.5,
        "end": 2.5,
        "text": "hello",
        "tokens": [1, 2, 3],
        "temperature": 0.0,
        "avg_logprob": -0.1,
        "compression_ratio": 1.2,
        "no_speech_prob": 0.01,
    }


def test_verbose_json_omits_words_when_not_requested():
    """未要求字級時間戳時，verbose_json 不輸出 words 欄位。"""
    client = _client([_segment("hello", words=[_word("hello", 0.0, 1.0)])])
    response = client.post(ENDPOINT, **_upload(response_format="verbose_json"))
    assert response.status_code == 200
    assert "words" not in response.json()


def test_timestamp_granularities_enables_word_timestamps():
    """timestamp_granularities 含 word 時輸出 words，並開啟引擎的 word_timestamps。"""
    captured = {}

    class WordEngine:
        async def transcribe_async_method(self, flow_id, audio_path, params):
            del flow_id, audio_path
            captured["word_timestamps"] = params.word_timestamps
            return [_segment("hello", words=[_word("hello", 0.2, 0.9)])], _info()

    client = _client_with_engine(WordEngine())
    response = client.post(
        ENDPOINT,
        files={"file": ("sample.wav", _minimal_wav(), "audio/wav")},
        data={"response_format": "verbose_json", "timestamp_granularities[]": ["word"]},
    )
    assert response.status_code == 200
    assert captured["word_timestamps"] is True
    assert response.json()["words"] == [{"word": "hello", "start": 0.2, "end": 0.9}]


def test_timestamp_granularities_without_brackets_is_accepted():
    """未帶方括號的 timestamp_granularities 欄位名同樣生效。"""
    client = _client([_segment("hello", words=[_word("hello", 0.0, 1.0)])])
    response = client.post(
        ENDPOINT,
        files={"file": ("sample.wav", _minimal_wav(), "audio/wav")},
        data={"response_format": "verbose_json", "timestamp_granularities": ["word"]},
    )
    assert response.status_code == 200
    assert response.json()["words"] == [{"word": "hello", "start": 0.0, "end": 1.0}]


def test_verbose_json_duration_uses_info_not_segment_sum():
    """duration 取原始音檔長度，不用片段時長總和，避免砍頭砍尾造成低估。"""
    client = _client(
        [_segment("hello", start=0.0, end=1.0)],
        info=_info(duration=30.0),
    )
    response = client.post(ENDPOINT, **_upload(response_format="verbose_json"))
    assert response.status_code == 200
    assert response.json()["duration"] == 30.0


def test_prompt_maps_to_initial_prompt():
    """OpenAI 的 prompt 映射為 faster-whisper 的 initial_prompt。"""
    client = _client_with_engine(
        EchoParamsEngine(lambda flow_id, params: params.initial_prompt or "")
    )
    response = client.post(ENDPOINT, **_upload(prompt="病歷用語"))
    assert response.status_code == 200
    assert response.json() == {"text": "病歷用語"}


def test_omitted_prompt_results_in_no_initial_prompt():
    """未傳 prompt 時 initial_prompt 為 None，不套用服務預設提示詞。"""
    client = _client_with_engine(
        EchoParamsEngine(lambda flow_id, params: str(params.initial_prompt))
    )
    response = client.post(ENDPOINT, **_upload())
    assert response.status_code == 200
    assert response.json() == {"text": "None"}


def test_language_is_normalized():
    """language 經 normalize_language 轉小寫並去空白。"""
    client = _client_with_engine(
        EchoParamsEngine(lambda flow_id, params: str(params.language))
    )
    response = client.post(ENDPOINT, **_upload(language="  ZH  "))
    assert response.status_code == 200
    assert response.json() == {"text": "zh"}


def test_blank_language_means_auto_detect():
    """空字串 language 視為自動偵測，傳給引擎的是 None。"""
    client = _client_with_engine(
        EchoParamsEngine(lambda flow_id, params: str(params.language))
    )
    response = client.post(ENDPOINT, **_upload(language=""))
    assert response.status_code == 200
    assert response.json() == {"text": "None"}


def test_extra_body_params_are_coerced_and_forwarded():
    """extra_body 帶入的 faster-whisper 參數會轉型並傳到引擎。"""

    def formatter(flow_id, params):
        del flow_id
        return (
            f"{params.beam_size}|{params.enable_vad}|{params.use_fast_pipeline}"
            f"|{params.repetition_penalty}|{params.hotwords}"
        )

    client = _client_with_engine(EchoParamsEngine(formatter))
    response = client.post(
        ENDPOINT,
        **_upload(
            beam_size="3",
            enable_vad="false",
            use_fast_pipeline="true",
            repetition_penalty="1.2",
            hotwords="續觀,依醫囑",
        ),
    )
    assert response.status_code == 200
    assert response.json() == {"text": "3|False|True|1.2|續觀,依醫囑"}


def test_vad_params_are_forwarded():
    """VAD 參數經 extra_body 傳入後會轉型並傳到引擎。"""

    def formatter(flow_id, params):
        del flow_id
        return (
            f"{params.vad_filter_ms}|{params.vad_threshold}|{params.neg_threshold}"
            f"|{params.min_speech_duration_ms}|{params.max_speech_duration_s}|{params.speech_pad_ms}"
        )

    client = _client_with_engine(EchoParamsEngine(formatter))
    response = client.post(
        ENDPOINT,
        **_upload(
            vad_filter_ms="800",
            vad_threshold="0.4",
            neg_threshold="0.2",
            min_speech_duration_ms="150",
            max_speech_duration_s="20.5",
            speech_pad_ms="100",
        ),
    )
    assert response.status_code == 200
    assert response.json() == {"text": "800|0.4|0.2|150|20.5|100"}


def test_omitted_beam_size_uses_env(monkeypatch):
    """未傳 beam_size 時沿用 WHISPER_BEAM_SIZE。"""
    monkeypatch.setenv("WHISPER_BEAM_SIZE", "2")
    client = _client_with_engine(
        EchoParamsEngine(lambda flow_id, params: str(params.beam_size))
    )
    response = client.post(ENDPOINT, **_upload())
    assert response.status_code == 200
    assert response.json() == {"text": "2"}


def test_temperature_is_accepted_but_ignored():
    """temperature 收下後忽略，不影響轉錄結果也不回錯誤。"""
    client = _client([_segment("hello")])
    with_temperature = client.post(ENDPOINT, **_upload(temperature="0.8"))
    without_temperature = client.post(ENDPOINT, **_upload())
    assert with_temperature.status_code == 200
    assert with_temperature.json() == without_temperature.json() == {"text": "hello"}


def test_unknown_model_falls_back_to_defaults():
    """未知 model 名稱仍正常轉錄，不回 404。"""
    client = _client([_segment("hello")])
    response = client.post(ENDPOINT, **_upload(model="some-random-deployment-name"))
    assert response.status_code == 200
    assert response.json() == {"text": "hello"}


def test_x_request_id_is_used_as_flow_id():
    """帶 X-Request-Id 時以其值作為 flow_id，供 trace 串接。"""
    client = _client_with_engine(EchoParamsEngine(lambda flow_id, params: flow_id))
    response = client.post(
        ENDPOINT,
        files={"file": ("sample.wav", _minimal_wav(), "audio/wav")},
        headers={"X-Request-Id": "upstream-trace-1"},
    )
    assert response.status_code == 200
    assert response.json() == {"text": "upstream-trace-1"}


def test_simplified_chinese_is_converted_to_traditional():
    """轉錄文字經 OpenCC s2tw 轉為繁體。"""
    client = _client([_segment("这是简体")])
    response = client.post(ENDPOINT, **_upload())
    assert response.status_code == 200
    assert response.json() == {"text": "這是簡體"}


def test_verbose_json_segments_are_converted_to_traditional():
    """verbose_json 的 segment 文字與 text 同步做簡繁轉換，兩者串接結果一致。"""
    client = _client([_segment("这是简体")])
    response = client.post(ENDPOINT, **_upload(response_format="verbose_json"))
    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "這是簡體"
    assert body["segments"][0]["text"] == "這是簡體"


def test_blacklist_segment_is_dropped():
    """命中幻覺 blacklist 的 segment 不會出現在回應。"""
    client = _client([_segment("歡迎訂閱"), _segment("真實內容", seg_id=1)])
    response = client.post(ENDPOINT, **_upload())
    assert response.status_code == 200
    assert response.json() == {"text": "真實內容"}


def test_unsupported_response_format_returns_error_envelope():
    """不支援的 response_format 回 400 且為 OpenAI 錯誤信封。"""
    client = _client()
    response = client.post(ENDPOINT, **_upload(response_format="srt"))
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == "response_format"
    assert "srt" in error["message"]


def test_stream_true_returns_error_envelope():
    """stream=true 回 400，whisper 不支援串流轉錄。"""
    client = _client()
    response = client.post(ENDPOINT, **_upload(stream="true"))
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "stream"


def test_missing_file_returns_error_envelope():
    """缺少 file 欄位時回 400 且為 OpenAI 錯誤信封，而非 FastAPI 預設 422。"""
    client = _client()
    response = client.post(ENDPOINT, data={"model": "whisper-1"})
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == "file"


def test_empty_file_returns_error_envelope():
    """上傳空檔時回 400 且為 OpenAI 錯誤信封。"""
    client = _client()
    response = client.post(
        ENDPOINT,
        files={"file": ("sample.wav", b"", "audio/wav")},
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "MSS-VTT-000701"
    assert error["param"] == "file"


def test_validation_handler_leaves_non_v1_paths_on_fastapi_default():
    """驗證錯誤處理只改寫 /v1；/audio/* 仍回 FastAPI 預設的 422 detail 格式。"""
    app = FastAPI()
    app.include_router(router, prefix="/v1")
    app.add_exception_handler(OpenAICompatError, openai_error_handler)
    app.add_exception_handler(RequestValidationError, openai_validation_error_handler)
    service = TranscriptionService(engine=FakeEngine([_segment("hello")]))
    app.dependency_overrides[get_openai_transcription_service] = lambda: service

    @app.post("/audio/legacy/")
    async def legacy(required_param: int):
        return {"ok": required_param}

    client = TestClient(app)

    legacy_response = client.post("/audio/legacy/")
    assert legacy_response.status_code == 422
    assert "detail" in legacy_response.json()
    assert "error" not in legacy_response.json()

    openai_response = client.post(ENDPOINT, data={"model": "whisper-1"})
    assert openai_response.status_code == 400
    assert "error" in openai_response.json()


def test_engine_failure_returns_server_error_envelope():
    """引擎拋例外時回 500 且為 OpenAI 錯誤信封。"""

    class BrokenEngine:
        async def transcribe_async_method(self, flow_id, audio_path, params):
            del flow_id, audio_path, params
            raise RuntimeError("decode exploded")

    client = _client_with_engine(BrokenEngine())
    response = client.post(ENDPOINT, **_upload())
    assert response.status_code == 500
    error = response.json()["error"]
    assert error["type"] == "server_error"
    assert error["code"] == "MSS-VTT-000000"


class FakeDiarizationEngine:
    """回傳固定 WhisperX 片段的語者分離引擎。"""

    def __init__(self, wx_segments, *, is_enable: bool = True):
        """保存預設語者片段。

        Args:
            wx_segments: diarize_to_segments 回傳的片段。
            is_enable: 引擎是否可用。
        """
        self.is_enable = is_enable
        self._wx_segments = wx_segments
        self.last_params = None

    def diarize_to_segments(self, flow_id, segments, info, audio_path, diarization_params):
        """回傳建構時給定的 WhisperX 片段。

        Args:
            flow_id: 流程識別，測試忽略。
            segments: 轉錄片段，測試忽略。
            info: 轉錄資訊，測試忽略。
            audio_path: 音檔路徑，測試忽略。
            diarization_params: 語者分離參數。

        Returns:
            WhisperX 片段清單。
        """
        del flow_id, segments, info, audio_path
        self.last_params = diarization_params
        return list(self._wx_segments)

    def merge_speaker_blocks(self, segments):
        """將連續同一語者片段合併為 Speaker block。

        Args:
            segments: 含 speaker／start／end／text 的片段。

        Returns:
            合併後的 Speaker block 清單。
        """
        blocks = []
        block_speaker = None
        block_start = None
        block_end = None
        block_texts = []

        for segment in segments:
            text = (segment.get("text") or "").strip()
            if not text:
                continue
            speaker_key = segment.get("speaker") or "UNKNOWN"
            start = segment.get("start", 0.0)
            end = segment.get("end", 0.0)
            if speaker_key != block_speaker:
                if block_speaker is not None and block_texts:
                    blocks.append(
                        {
                            "speaker": block_speaker,
                            "start": block_start,
                            "end": block_end,
                            "text": " ".join(block_texts),
                        }
                    )
                block_speaker = speaker_key
                block_start = start
                block_texts = []
            block_end = end
            block_texts.append(text)

        if block_speaker is not None and block_texts:
            blocks.append(
                {
                    "speaker": block_speaker,
                    "start": block_start,
                    "end": block_end,
                    "text": " ".join(block_texts),
                }
            )
        return blocks


def test_speaker_flags_with_json_return_response_format_error():
    """json + diarization／always_speaker_label 回 400，param 為 response_format。"""
    client = _client()
    for flag in ("diarization", "always_speaker_label"):
        response = client.post(ENDPOINT, **_upload(**{flag: "true"}))
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["param"] == "response_format"
        assert error["code"] == "MSS-VTT-000102"


def test_speaker_flags_with_text_return_response_format_error():
    """text + 語者旗標回 400，param 為 response_format。"""
    client = _client()
    response = client.post(
        ENDPOINT,
        **_upload(response_format="text", always_speaker_label="true"),
    )
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "response_format"


def test_diarization_unavailable_returns_error():
    """diarization=true 但引擎未載入時回 400，param 為 diarization。"""
    client = _client_with_engine(FakeEngine([_segment("hello")]), diarization_engine=None)
    response = client.post(
        ENDPOINT,
        **_upload(response_format="verbose_json", diarization="true"),
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["param"] == "diarization"
    assert error["code"] == "MSS-VTT-000102"


def test_diarization_disabled_engine_returns_error():
    """diarization=true 但 is_enable=False 時回 400。"""
    disabled = FakeDiarizationEngine([], is_enable=False)
    client = _client_with_engine(FakeEngine([_segment("hello")]), diarization_engine=disabled)
    response = client.post(
        ENDPOINT,
        **_upload(response_format="verbose_json", diarization="true"),
    )
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "diarization"


def test_always_speaker_label_stamps_verbose_json_without_prefixing_text():
    """always_speaker_label 在 segments 蓋 SPEAKER_00，text 仍是純語音。"""
    client = _client([_segment("大家好", start=0.0, end=1.5)])
    response = client.post(
        ENDPOINT,
        **_upload(response_format="verbose_json", always_speaker_label="true"),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "大家好"
    assert body["segments"][0]["speaker"] == "SPEAKER_00"
    assert "[" not in body["text"]


def test_diarization_merges_speaker_blocks_without_word_timestamps():
    """未要求字級時，verbose_json segment 為合併後的 Speaker block。"""
    wx_segments = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0, "text": "大家好"},
        {"speaker": "SPEAKER_00", "start": 1.0, "end": 2.0, "text": "我是"},
        {"speaker": "SPEAKER_01", "start": 2.0, "end": 3.5, "text": "小明"},
    ]
    dia = FakeDiarizationEngine(wx_segments)
    client = _client_with_engine(FakeEngine([_segment("ignored")]), diarization_engine=dia)
    response = client.post(
        ENDPOINT,
        **_upload(response_format="verbose_json", diarization="true"),
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["segments"]) == 2
    assert body["segments"][0]["speaker"] == "SPEAKER_00"
    assert body["segments"][0]["text"] == "大家好 我是"
    assert body["segments"][0]["start"] == 0.0
    assert body["segments"][0]["end"] == 2.0
    assert body["segments"][1]["speaker"] == "SPEAKER_01"
    assert body["text"] == "大家好 我是小明"
    assert "words" not in body


def test_diarization_keeps_aligned_segments_with_word_timestamps():
    """要求字級時不合塊，speaker 在字上。"""
    wx_segments = [
        {
            "speaker": "SPEAKER_00",
            "start": 0.0,
            "end": 1.0,
            "text": "大家好",
            "words": [
                {"word": "大", "start": 0.0, "end": 0.4, "speaker": "SPEAKER_00"},
                {"word": "家好", "start": 0.4, "end": 1.0, "speaker": "SPEAKER_00"},
            ],
        },
        {
            "speaker": "SPEAKER_00",
            "start": 1.0,
            "end": 2.0,
            "text": "我是",
            "words": [
                {"word": "我", "start": 1.0, "end": 1.4, "speaker": "SPEAKER_00"},
                {"word": "是", "start": 1.4, "end": 2.0, "speaker": "SPEAKER_00"},
            ],
        },
    ]
    dia = FakeDiarizationEngine(wx_segments)
    client = _client_with_engine(FakeEngine([_segment("ignored")]), diarization_engine=dia)
    response = client.post(
        ENDPOINT,
        **_upload(
            response_format="verbose_json",
            diarization="true",
            word_timestamps="true",
        ),
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["segments"]) == 2
    assert body["segments"][0]["speaker"] == "SPEAKER_00"
    assert body["segments"][0]["text"] == "大家好"
    assert len(body["words"]) == 4
    assert body["words"][0]["speaker"] == "SPEAKER_00"


def test_min_speaker_without_diarization_keeps_json_ok():
    """只傳 min_speaker、未開 diarization 時 json 仍 200，不觸發語者契約。"""
    client = _client([_segment("hello")])
    response = client.post(ENDPOINT, **_upload(min_speaker=2))
    assert response.status_code == 200
    assert response.json() == {"text": "hello"}


def test_verbose_json_without_speaker_flags_omits_speaker_field():
    """未帶語者旗標時 verbose_json 不含 speaker 欄位。"""
    client = _client([_segment("hello")])
    response = client.post(ENDPOINT, **_upload(response_format="verbose_json"))
    assert response.status_code == 200
    body = response.json()
    assert "speaker" not in body["segments"][0]


def test_diarization_passes_min_max_speaker_params():
    """diarization=true 時把 min_speaker／max_speaker 傳入 DiarizationParams。"""
    dia = FakeDiarizationEngine(
        [{"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0, "text": "hi"}]
    )
    client = _client_with_engine(FakeEngine([_segment("hi")]), diarization_engine=dia)
    response = client.post(
        ENDPOINT,
        **_upload(
            response_format="verbose_json",
            diarization="true",
            min_speaker=1,
            max_speaker=3,
        ),
    )
    assert response.status_code == 200
    assert dia.last_params.diarization is True
    assert dia.last_params.min_speaker == 1
    assert dia.last_params.max_speaker == 3
