import importlib
import sys


def test_load_without_diarization_uses_whisper_model_folder_from_env(monkeypatch):
    """未開 diarization 時可載入，且引擎使用 WHISPER_MODEL_FOLDER。"""
    monkeypatch.setenv("ENABLE_DIARIZATION", "false")
    monkeypatch.setenv("WHISPER_MODEL_FOLDER", "/models/realtime")
    original_loader = sys.modules.get("transcription.engine_loader")
    monkeypatch.setitem(sys.modules, "transcription.diarization", None)
    sys.modules.pop("transcription.engine_loader", None)

    try:
        engine_loader = importlib.import_module("transcription.engine_loader")

        class RecordingEngine:
            def __init__(self, model_size: str):
                self.model_size = model_size

        engine_loader.FasterWhisperEngine = RecordingEngine
        engine_loader.load()

        assert engine_loader.get_faster_whisper_engine().model_size == "/models/realtime"
        assert engine_loader.get_diarization_engine() is None
    finally:
        if original_loader is not None:
            sys.modules["transcription.engine_loader"] = original_loader
        else:
            sys.modules.pop("transcription.engine_loader", None)


def test_transcription_service_imports_without_diarization_module(monkeypatch):
    """chunk 解碼路徑在沒有 diarization 模組時仍可載入 TranscriptionService。"""
    original_service = sys.modules.get("transcription.service")
    monkeypatch.setitem(sys.modules, "transcription.diarization", None)
    sys.modules.pop("transcription.service", None)

    try:
        importlib.import_module("transcription.service")
    finally:
        if original_service is not None:
            sys.modules["transcription.service"] = original_service
        else:
            sys.modules.pop("transcription.service", None)
