from transcription.engine_loader import (
    get_diarization_engine,
    get_faster_whisper_engine,
    get_vad_engine as get_loaded_vad_engine,
)
from transcription.service import TranscriptionService
from vad.engine import SileroVadEngine


def get_openai_transcription_service() -> TranscriptionService:
    """組裝含 diarization 的 TranscriptionService，供 OpenAI 相容端點 Depends 使用。"""
    return TranscriptionService(
        engine=get_faster_whisper_engine(),
        diarization_engine=get_diarization_engine(),
    )


def get_vad_engine() -> SileroVadEngine:
    """回傳啟動時載入的 SileroVadEngine，供 vad_router Depends 使用。"""
    return get_loaded_vad_engine()
