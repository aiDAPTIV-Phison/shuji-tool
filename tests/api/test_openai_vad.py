import io
import wave

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from api.dependencies import get_vad_engine
from api.openai_errors import (
    OpenAICompatError,
    openai_error_handler,
    openai_validation_error_handler,
)
from api.openai_router import router

ENDPOINT = "/v1/audio/vad"


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


class FakeVadEngine:
    """回傳可控 Speech timestamps，並記錄收到的參數。"""

    def __init__(self, timestamps=None):
        """保存預設時段。

        Args:
            timestamps: 回傳的 start_ms／end_ms 列表；省略則為空列表。
        """
        self._timestamps = timestamps if timestamps is not None else []
        self.last_params = None

    async def detect_speech_timestamps(self, flow_id, audio_path, params):
        """記錄 params 並回傳預設時段。

        Args:
            flow_id: 流程識別，測試忽略。
            audio_path: 暫存 wav 路徑，測試忽略。
            params: 路由組出的偵測參數。

        Returns:
            Speech timestamps 列表。
        """
        del flow_id, audio_path
        self.last_params = params
        return list(self._timestamps)


def _client(engine=None) -> tuple[TestClient, FakeVadEngine]:
    """組出掛 OpenAI VAD 路由並注入 FakeVadEngine 的 TestClient。

    Args:
        engine: 假引擎；省略則回空 timestamps。

    Returns:
        (TestClient, FakeVadEngine)。
    """
    app = FastAPI()
    app.include_router(router, prefix="/v1")
    app.add_exception_handler(OpenAICompatError, openai_error_handler)
    app.add_exception_handler(RequestValidationError, openai_validation_error_handler)
    fake = engine or FakeVadEngine()
    app.dependency_overrides[get_vad_engine] = lambda: fake
    return TestClient(app), fake


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
    client, _ = _client()
    response = client.post(ENDPOINT, **_upload())
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "MSS-VTT-000101"
    assert error["param"] == "flow_id"


def test_sequence_no_is_optional():
    """未傳 sequence_no 時仍可偵測。"""
    client, _ = _client(FakeVadEngine([{"start_ms": 0, "end_ms": 100}]))
    response = client.post(ENDPOINT, **_upload(flow_id="f1"))
    assert response.status_code == 200
    assert response.json() == {"speech_timestamps": [{"start_ms": 0, "end_ms": 100}]}


def test_empty_file_returns_400_with_000701():
    """空檔時回 400 且 code 為 MSS-VTT-000701。"""
    client, _ = _client()
    response = client.post(
        ENDPOINT,
        files={"file": ("empty.wav", b"", "audio/wav")},
        data={"flow_id": "f1"},
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "MSS-VTT-000701"
    assert error["param"] == "file"


def test_non_wav_file_returns_400_with_000701():
    """非 RIFF/WAVE 檔時回 400 且 code 為 MSS-VTT-000701。"""
    client, _ = _client()
    response = client.post(
        ENDPOINT,
        files={"file": ("bad.mp3", b"ID3\x04not-a-wav", "audio/mpeg")},
        data={"flow_id": "f1"},
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "MSS-VTT-000701"
    assert error["param"] == "file"


def test_valid_wav_returns_speech_timestamps():
    """合法 wav 回 200 且僅含 speech_timestamps。"""
    client, _ = _client(
        FakeVadEngine(
            [{"start_ms": 500, "end_ms": 1200}, {"start_ms": 2000, "end_ms": 2500}]
        )
    )
    response = client.post(ENDPOINT, **_upload(flow_id="f1", sequence_no=0))
    assert response.status_code == 200
    assert response.json() == {
        "speech_timestamps": [
            {"start_ms": 500, "end_ms": 1200},
            {"start_ms": 2000, "end_ms": 2500},
        ]
    }


def test_no_speech_returns_empty_list():
    """無語音時仍 200，speech_timestamps 為空陣列。"""
    client, _ = _client(FakeVadEngine([]))
    response = client.post(ENDPOINT, **_upload(flow_id="f1"))
    assert response.status_code == 200
    assert response.json() == {"speech_timestamps": []}


def test_omitted_knobs_are_not_filled_with_whisper_defaults():
    """未傳旋鈕時不套 WhisperParams 批次數字，全為 None。"""
    client, fake = _client()
    response = client.post(ENDPOINT, **_upload(flow_id="f1"))
    assert response.status_code == 200
    assert fake.last_params.vad_threshold is None
    assert fake.last_params.vad_filter_ms is None
    assert fake.last_params.neg_threshold is None
    assert fake.last_params.min_speech_duration_ms is None
    assert fake.last_params.max_speech_duration_s is None
    assert fake.last_params.speech_pad_ms is None


def test_optional_knobs_are_forwarded_to_engine():
    """有傳的選填旋鈕會轉給引擎。"""
    client, fake = _client()
    response = client.post(
        ENDPOINT,
        **_upload(
            flow_id="f1",
            vad_threshold=0.6,
            vad_filter_ms=300,
            neg_threshold=0.3,
            min_speech_duration_ms=50,
            max_speech_duration_s=10.0,
            speech_pad_ms=20,
        ),
    )
    assert response.status_code == 200
    assert fake.last_params.vad_threshold == 0.6
    assert fake.last_params.vad_filter_ms == 300
    assert fake.last_params.neg_threshold == 0.3
    assert fake.last_params.min_speech_duration_ms == 50
    assert fake.last_params.max_speech_duration_s == 10.0
    assert fake.last_params.speech_pad_ms == 20
