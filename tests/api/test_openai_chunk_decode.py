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

ENDPOINT = "/v1/audio/chunk_decode"


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


def _segment(
    text: str,
    no_speech_prob: float = 0.99,
    avg_logprob: float = -2.0,
    start: float = 0.0,
    end: float = 0.3,
):
    """組出測試用 segment（duck typing，不載入 faster-whisper）。

    Args:
        text: 片段文字。
        no_speech_prob: 無語音機率，預設高到足以觸發批次砍頭規則。
        avg_logprob: 平均 log 機率。
        start: 起始秒。
        end: 結束秒。

    Returns:
        帶轉錄欄位的 SimpleNamespace。
    """
    return SimpleNamespace(
        text=text,
        no_speech_prob=no_speech_prob,
        avg_logprob=avg_logprob,
        start=start,
        end=end,
        compression_ratio=1.0,
        temperature=0.0,
    )


class FakeEngine:
    """依 WhisperParams 回傳可控 segment，供 HTTP 測試注入。"""

    def __init__(self, segments):
        """保存預設解碼片段。

        Args:
            segments: 未帶 initial_prompt 且未開 VAD 時回傳的片段。
        """
        self._segments = segments

    async def transcribe_async_method(self, flow_id, audio_path, params):
        """依 params 決定回傳文字，用來觀察 VAD／延續上下文是否有傳到引擎。

        Args:
            flow_id: 流程識別，測試忽略。
            audio_path: 暫存 wav 路徑，測試忽略。
            params: 路由組出的轉錄參數。

        Returns:
            (segments, info) 與真實引擎相同的元組形狀。
        """
        del flow_id, audio_path
        if params.enable_vad:
            text = "vad-on"
        elif params.initial_prompt:
            text = params.initial_prompt
        else:
            text = self._segments[0].text if self._segments else ""
            if self._segments:
                return self._segments, SimpleNamespace(language="zh", language_probability=1.0)
            return [], SimpleNamespace(language="zh", language_probability=1.0)
        return [_segment(text)], SimpleNamespace(language="zh", language_probability=1.0)


def _client_with_engine(engine) -> TestClient:
    """組出掛 OpenAI chunk 解碼路由並注入自訂引擎的 TestClient。

    Args:
        engine: 轉錄引擎實例。

    Returns:
        FastAPI TestClient。
    """
    app = FastAPI()
    app.include_router(router, prefix="/v1")
    app.add_exception_handler(OpenAICompatError, openai_error_handler)
    app.add_exception_handler(RequestValidationError, openai_validation_error_handler)
    service = TranscriptionService(engine=engine)
    app.dependency_overrides[get_openai_transcription_service] = lambda: service
    return TestClient(app)


def _client(segments=None) -> TestClient:
    """組出注入 FakeEngine 的 TestClient。

    Args:
        segments: FakeEngine 預設片段；省略則為 `partial-ok`。

    Returns:
        FastAPI TestClient。
    """
    return _client_with_engine(FakeEngine(segments or [_segment("partial-ok")]))


class FakeDiarizationEngine:
    """回傳可控 turns／speakers，供 HTTP 測試注入。"""

    def __init__(self, turns, speakers, error=None):
        """保存 diarize_chunk 的回傳或例外。

        Args:
            turns: 講者片段列表。
            speakers: 本包講者與向量列表。
            error: 若設定則 diarize_chunk 拋出此例外。
        """
        self._turns = turns
        self._speakers = speakers
        self._error = error
        self.is_enable = True

    def diarize_chunk(self, flow_id, segments, info, audio_path):
        """回傳預設結構化結果，或依設定拋錯。

        Args:
            flow_id: 流程識別，測試忽略。
            segments: ASR 片段，測試忽略。
            info: 轉錄資訊，測試忽略。
            audio_path: 暫存 wav 路徑，測試忽略。

        Returns:
            含 turns、speakers 的 SimpleNamespace。
        """
        del flow_id, segments, info, audio_path
        if self._error is not None:
            raise self._error
        return SimpleNamespace(turns=self._turns, speakers=self._speakers)


def _client_with_diarization(segments=None, diarization_engine=None) -> TestClient:
    """組出掛 OpenAI chunk 解碼路由並注入 ASR 與 diarization 假引擎的 TestClient。

    Args:
        segments: FakeEngine 預設片段；省略則為 `大家好接下來換我說`。
        diarization_engine: 假 diarization 引擎；省略則為 None。

    Returns:
        FastAPI TestClient。
    """
    app = FastAPI()
    app.include_router(router, prefix="/v1")
    app.add_exception_handler(OpenAICompatError, openai_error_handler)
    app.add_exception_handler(RequestValidationError, openai_validation_error_handler)
    service = TranscriptionService(
        engine=FakeEngine(segments or [_segment("大家好接下來換我說")]),
        diarization_engine=diarization_engine,
    )
    app.dependency_overrides[get_openai_transcription_service] = lambda: service
    return TestClient(app)


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


def test_missing_flow_id_returns_400_openai_envelope():
    """缺少必填 flow_id 時回 400 OpenAI 錯誤信封。"""
    client = _client()
    response = client.post(ENDPOINT, **_upload(sequence_no=0))
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "MSS-VTT-000101"
    assert error["param"] == "flow_id"


def test_missing_sequence_no_returns_400_openai_envelope():
    """缺少必填 sequence_no 時回 400 OpenAI 錯誤信封。"""
    client = _client()
    response = client.post(ENDPOINT, **_upload(flow_id="f1"))
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "MSS-VTT-000101"
    assert error["param"] == "sequence_no"


def test_negative_sequence_no_returns_400_openai_envelope():
    """sequence_no 小於 0 時回 400 OpenAI 錯誤信封。"""
    client = _client()
    response = client.post(ENDPOINT, **_upload(flow_id="f1", sequence_no=-1))
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "MSS-VTT-000101"


def test_empty_file_returns_400_with_000701():
    """空檔時回 400 且 code 為 MSS-VTT-000701。"""
    client = _client()
    response = client.post(
        ENDPOINT,
        files={"file": ("empty.wav", b"", "audio/wav")},
        data={"flow_id": "f1", "sequence_no": 0},
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "MSS-VTT-000701"
    assert error["param"] == "file"


def test_non_wav_file_returns_400_with_000701():
    """非 RIFF/WAVE 檔時回 400 且 code 為 MSS-VTT-000701。"""
    client = _client()
    response = client.post(
        ENDPOINT,
        files={"file": ("bad.mp3", b"ID3\x04not-a-wav", "audio/mpeg")},
        data={"flow_id": "f1", "sequence_no": 0},
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "MSS-VTT-000701"
    assert error["param"] == "file"


def test_valid_wav_returns_text():
    """合法 wav 回 200 且僅含 text 欄位。"""
    client = _client([_segment("partial-ok")])
    response = client.post(ENDPOINT, **_upload(flow_id="f1", sequence_no=0))
    assert response.status_code == 200
    assert response.json() == {"text": "partial-ok"}


def test_omitted_is_final_returns_200():
    """未傳 is_final 時仍可解碼（選填，僅供 log）。"""
    client = _client([_segment("partial-ok")])
    response = client.post(ENDPOINT, **_upload(flow_id="f1", sequence_no=0))
    assert response.status_code == 200
    assert response.json() == {"text": "partial-ok"}


def test_is_final_does_not_change_decode_result():
    """is_final 僅供 log，不改變解碼結果。"""
    client = _client([_segment("same-text")])
    without_flag = client.post(ENDPOINT, **_upload(flow_id="f1", sequence_no=0))
    with_flag = client.post(
        ENDPOINT, **_upload(flow_id="f1", sequence_no=0, is_final=True)
    )
    assert without_flag.status_code == 200
    assert with_flag.status_code == 200
    assert without_flag.json() == with_flag.json() == {"text": "same-text"}


def test_initial_prompt_is_used_as_continuation_context():
    """帶 initial_prompt 時解碼結果銜接該前文。"""
    client = _client()
    response = client.post(
        ENDPOINT,
        **_upload(
            flow_id="f1",
            sequence_no=1,
            initial_prompt="先前的逐字稿",
            condition_on_previous_text=True,
            language="zh",
        ),
    )
    assert response.status_code == 200
    assert response.json() == {"text": "先前的逐字稿"}


def test_whisper_vad_is_disabled():
    """chunk 解碼關閉 Whisper 內建 VAD，不會走 vad-on 分支。"""
    client = _client()
    response = client.post(ENDPOINT, **_upload(flow_id="f1", sequence_no=0))
    assert response.status_code == 200
    assert response.json()["text"] != "vad-on"


def test_simplified_chinese_is_converted_to_traditional():
    """解碼文字經 OpenCC s2tw 轉為繁體。"""
    client = _client([_segment("这是简体")])
    response = client.post(ENDPOINT, **_upload(flow_id="f1", sequence_no=0))
    assert response.status_code == 200
    assert response.json() == {"text": "這是簡體"}


def test_blacklist_segment_is_dropped():
    """命中幻覺 blacklist 的 segment 不會出現在 text。"""
    client = _client([_segment("歡迎訂閱"), _segment("真實內容")])
    response = client.post(ENDPOINT, **_upload(flow_id="f1", sequence_no=0))
    assert response.status_code == 200
    assert response.json() == {"text": "真實內容"}


def test_high_no_speech_segment_is_not_trimmed_as_file_edges():
    """短 chunk 不套用批次砍頭砍尾，高 no_speech 片段仍保留。"""
    client = _client([_segment("keep me", no_speech_prob=0.99, avg_logprob=-2.0, end=0.3)])
    response = client.post(ENDPOINT, **_upload(flow_id="f1", sequence_no=2))
    assert response.status_code == 200
    assert response.json() == {"text": "keep me"}


def test_beam_size_form_overrides_env(monkeypatch):
    """beam_size form 覆寫 WHISPER_BEAM_SIZE 並傳到引擎。"""
    monkeypatch.setenv("WHISPER_BEAM_SIZE", "1")

    class EchoBeamEngine:
        async def transcribe_async_method(self, flow_id, audio_path, params):
            del flow_id, audio_path
            return (
                [_segment(str(params.beam_size))],
                SimpleNamespace(language="zh", language_probability=1.0),
            )

    client = _client_with_engine(EchoBeamEngine())
    response = client.post(
        ENDPOINT, **_upload(flow_id="f1", sequence_no=0, beam_size=3)
    )
    assert response.status_code == 200
    assert response.json() == {"text": "3"}


def test_omitted_beam_size_uses_env(monkeypatch):
    """未傳 beam_size 時使用 WHISPER_BEAM_SIZE。"""
    monkeypatch.setenv("WHISPER_BEAM_SIZE", "1")

    class EchoBeamEngine:
        async def transcribe_async_method(self, flow_id, audio_path, params):
            del flow_id, audio_path
            return (
                [_segment(str(params.beam_size))],
                SimpleNamespace(language="zh", language_probability=1.0),
            )

    client = _client_with_engine(EchoBeamEngine())
    response = client.post(ENDPOINT, **_upload(flow_id="f1", sequence_no=0))
    assert response.status_code == 200
    assert response.json() == {"text": "1"}


def test_hotwords_form_is_used():
    """hotwords form 會傳到引擎。"""

    class EchoHotwordsEngine:
        async def transcribe_async_method(self, flow_id, audio_path, params):
            del flow_id, audio_path
            return (
                [_segment(params.hotwords or "")],
                SimpleNamespace(language="zh", language_probability=1.0),
            )

    client = _client_with_engine(EchoHotwordsEngine())
    response = client.post(
        ENDPOINT, **_upload(flow_id="f1", sequence_no=0, hotwords="專有名詞")
    )
    assert response.status_code == 200
    assert response.json() == {"text": "專有名詞"}


def test_use_fast_pipeline_form_is_used():
    """use_fast_pipeline form 會傳到引擎。"""

    class EchoPipelineEngine:
        async def transcribe_async_method(self, flow_id, audio_path, params):
            del flow_id, audio_path
            return (
                [_segment("fast" if params.use_fast_pipeline else "slow")],
                SimpleNamespace(language="zh", language_probability=1.0),
            )

    client = _client_with_engine(EchoPipelineEngine())
    response = client.post(
        ENDPOINT, **_upload(flow_id="f1", sequence_no=0, use_fast_pipeline=True)
    )
    assert response.status_code == 200
    assert response.json() == {"text": "fast"}


def test_omitted_use_fast_pipeline_defaults_to_false():
    """未傳 use_fast_pipeline 時使用 process 預設 False。"""

    class EchoPipelineEngine:
        async def transcribe_async_method(self, flow_id, audio_path, params):
            del flow_id, audio_path
            return (
                [_segment("fast" if params.use_fast_pipeline else "slow")],
                SimpleNamespace(language="zh", language_probability=1.0),
            )

    client = _client_with_engine(EchoPipelineEngine())
    response = client.post(ENDPOINT, **_upload(flow_id="f1", sequence_no=0))
    assert response.status_code == 200
    assert response.json() == {"text": "slow"}


def test_repetition_penalty_form_is_used():
    """repetition_penalty form 會傳到引擎。"""

    class EchoPenaltyEngine:
        async def transcribe_async_method(self, flow_id, audio_path, params):
            del flow_id, audio_path
            return (
                [_segment(str(params.repetition_penalty))],
                SimpleNamespace(language="zh", language_probability=1.0),
            )

    client = _client_with_engine(EchoPenaltyEngine())
    response = client.post(
        ENDPOINT, **_upload(flow_id="f1", sequence_no=0, repetition_penalty=1.2)
    )
    assert response.status_code == 200
    assert response.json() == {"text": "1.2"}


def test_omitted_repetition_penalty_defaults_to_one():
    """未傳 repetition_penalty 時使用 process 預設 1.0。"""

    class EchoPenaltyEngine:
        async def transcribe_async_method(self, flow_id, audio_path, params):
            del flow_id, audio_path
            return (
                [_segment(str(params.repetition_penalty))],
                SimpleNamespace(language="zh", language_probability=1.0),
            )

    client = _client_with_engine(EchoPenaltyEngine())
    response = client.post(ENDPOINT, **_upload(flow_id="f1", sequence_no=0))
    assert response.status_code == 200
    assert response.json() == {"text": "1.0"}


FINAL_ENDPOINT = "/v1/audio/chunk_decode_final"


def test_diarization_true_returns_turns_and_speakers():
    """diarization=true 且引擎成功時回 200，含 text、turns 與 speakers。"""
    diarization_engine = FakeDiarizationEngine(
        turns=[
            SimpleNamespace(text="大家好", local_speaker="SPEAKER_00"),
            SimpleNamespace(text="接下來換我說", local_speaker="SPEAKER_01"),
        ],
        speakers=[
            SimpleNamespace(local_speaker="SPEAKER_00", embeddings=[0.12, -0.04]),
            SimpleNamespace(local_speaker="SPEAKER_01", embeddings=[0.08, 0.33]),
        ],
    )
    client = _client_with_diarization(diarization_engine=diarization_engine)
    response = client.post(
        ENDPOINT, **_upload(flow_id="f1", sequence_no=0, diarization="true")
    )
    assert response.status_code == 200
    assert response.json() == {
        "text": "大家好接下來換我說",
        "turns": [
            {"text": "大家好", "local_speaker": "SPEAKER_00"},
            {"text": "接下來換我說", "local_speaker": "SPEAKER_01"},
        ],
        "speakers": [
            {"local_speaker": "SPEAKER_00", "embeddings": [0.12, -0.04]},
            {"local_speaker": "SPEAKER_01", "embeddings": [0.08, 0.33]},
        ],
    }


def test_diarization_not_true_omits_turns_and_speakers():
    """省略或 false 時即使注入引擎也只回 text。"""
    diarization_engine = FakeDiarizationEngine(
        turns=[SimpleNamespace(text="大家好", local_speaker="SPEAKER_00")],
        speakers=[SimpleNamespace(local_speaker="SPEAKER_00", embeddings=[0.12, -0.04])],
    )
    client = _client_with_diarization(diarization_engine=diarization_engine)
    omitted = client.post(ENDPOINT, **_upload(flow_id="f1", sequence_no=0))
    not_true = client.post(
        ENDPOINT, **_upload(flow_id="f1", sequence_no=0, diarization="false")
    )
    assert omitted.status_code == 200
    assert not_true.status_code == 200
    assert omitted.json() == not_true.json() == {"text": "大家好接下來換我說"}


def test_diarization_true_without_engine_returns_text_only():
    """diarization=true 但未載入引擎時 200 且只有 text。"""
    client = _client_with_diarization(diarization_engine=None)
    response = client.post(
        ENDPOINT, **_upload(flow_id="f1", sequence_no=0, diarization="true")
    )
    assert response.status_code == 200
    assert response.json() == {"text": "大家好接下來換我說"}


def test_diarization_true_when_engine_disabled_returns_text_only():
    """diarization=true 但引擎 is_enable=False 時 200 且只有 text。"""
    diarization_engine = FakeDiarizationEngine(turns=[], speakers=[])
    diarization_engine.is_enable = False
    client = _client_with_diarization(diarization_engine=diarization_engine)
    response = client.post(
        ENDPOINT, **_upload(flow_id="f1", sequence_no=0, diarization="true")
    )
    assert response.status_code == 200
    assert response.json() == {"text": "大家好接下來換我說"}


def test_diarization_failure_returns_text_only():
    """ASR 成功但 diarize 失敗時 200 且只有 text。"""
    diarization_engine = FakeDiarizationEngine(
        turns=[],
        speakers=[],
        error=RuntimeError("diarize boom"),
    )
    client = _client_with_diarization(diarization_engine=diarization_engine)
    response = client.post(
        ENDPOINT, **_upload(flow_id="f1", sequence_no=0, diarization="true")
    )
    assert response.status_code == 200
    assert response.json() == {"text": "大家好接下來換我說"}


def test_diarization_true_with_no_speakers_returns_empty_arrays():
    """有稿但沒偵測到講者時 200，turns／speakers 為空陣列。"""
    diarization_engine = FakeDiarizationEngine(turns=[], speakers=[])
    client = _client_with_diarization(diarization_engine=diarization_engine)
    response = client.post(
        ENDPOINT, **_upload(flow_id="f1", sequence_no=0, diarization="true")
    )
    assert response.status_code == 200
    assert response.json() == {
        "text": "大家好接下來換我說",
        "turns": [],
        "speakers": [],
    }


def test_diarization_turn_text_is_converted_to_traditional():
    """turns 文字經 OpenCC s2tw 轉為繁體。"""
    diarization_engine = FakeDiarizationEngine(
        turns=[SimpleNamespace(text="这是简体", local_speaker="SPEAKER_00")],
        speakers=[SimpleNamespace(local_speaker="SPEAKER_00", embeddings=[0.1])],
    )
    client = _client_with_diarization(
        segments=[_segment("这是简体")],
        diarization_engine=diarization_engine,
    )
    response = client.post(
        ENDPOINT, **_upload(flow_id="f1", sequence_no=0, diarization="true")
    )
    assert response.status_code == 200
    assert response.json() == {
        "text": "這是簡體",
        "turns": [{"text": "這是簡體", "local_speaker": "SPEAKER_00"}],
        "speakers": [{"local_speaker": "SPEAKER_00", "embeddings": [0.1]}],
    }


def test_chunk_decode_final_diarization_true_returns_turns_and_speakers():
    """chunk_decode_final 與 chunk_decode 同一 diarization 契約。"""
    diarization_engine = FakeDiarizationEngine(
        turns=[SimpleNamespace(text="大家好", local_speaker="SPEAKER_00")],
        speakers=[SimpleNamespace(local_speaker="SPEAKER_00", embeddings=[0.12, -0.04])],
    )
    client = _client_with_diarization(
        segments=[_segment("大家好")],
        diarization_engine=diarization_engine,
    )
    response = client.post(
        FINAL_ENDPOINT, **_upload(flow_id="f1", sequence_no=0, diarization="true")
    )
    assert response.status_code == 200
    assert response.json() == {
        "text": "大家好",
        "turns": [{"text": "大家好", "local_speaker": "SPEAKER_00"}],
        "speakers": [{"local_speaker": "SPEAKER_00", "embeddings": [0.12, -0.04]}],
    }
