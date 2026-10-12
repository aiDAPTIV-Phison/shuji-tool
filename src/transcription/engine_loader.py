from __future__ import annotations

from typing import TYPE_CHECKING

from silero_vad import load_silero_vad

from transcription.engine import FasterWhisperEngine
from utils.ConfigHandler import ConfigHandler
from vad.engine import SileroVadEngine

if TYPE_CHECKING:
    from transcription.diarization import DiarizationEngine

_faster_whisper_engine: FasterWhisperEngine | None = None
_diarization_engine: DiarizationEngine | None = None
_vad_engine: SileroVadEngine | None = None


def load() -> None:
    """自 ConfigHandler 讀取路徑，建立並快取 FasterWhisperEngine 與 DiarizationEngine。"""
    global _faster_whisper_engine, _diarization_engine, _vad_engine

    # 已載入則略過，避免同 process 重複建立引擎（僅以 whisper 判斷，diarization 可能刻意為 None）
    if _faster_whisper_engine is not None:
        # whisper 已成功但 VAD 先前失敗時，允許補載 VAD
        if _vad_engine is None:
            _vad_engine = SileroVadEngine(model=load_silero_vad(onnx=True))
        return

    model_size = ConfigHandler.get_whisper_model_folder()
    _faster_whisper_engine = FasterWhisperEngine(model_size=model_size)

    # 僅在 ENABLE_DIARIZATION 開啟時載入 alignment 與 diarization
    if ConfigHandler.is_diarization_enabled():
        from transcription.diarization import AlignmentProfile, DiarizationEngine

        _diarization_engine = DiarizationEngine(
            alignment_profiles=[
                AlignmentProfile(
                    load_language_code="zh",
                    support_languages=["zh", "en"],
                    model_dir=ConfigHandler.get_alignment_model_folder(),
                )
            ],
            diarization_model_dir=ConfigHandler.get_diarization_model_folder(),
        )
    else:
        _diarization_engine = None

    # 即時 VAD：CPU ONNX，與 Whisper 解碼路徑分離
    _vad_engine = SileroVadEngine(model=load_silero_vad(onnx=True))


def get_faster_whisper_engine() -> FasterWhisperEngine:
    """回傳已載入的 FasterWhisperEngine。

    Raises:
        RuntimeError: 尚未呼叫 load() 時。
    """
    if _faster_whisper_engine is None:
        raise RuntimeError("FasterWhisperEngine 尚未載入，請先呼叫 model_loader.load()")
    return _faster_whisper_engine


def get_diarization_engine() -> DiarizationEngine | None:
    """回傳已載入的 DiarizationEngine；未啟用時為 None。

    Raises:
        RuntimeError: 尚未呼叫 load() 時。
    """
    if _faster_whisper_engine is None:
        raise RuntimeError("模型尚未載入，請先呼叫 model_loader.load()")
    return _diarization_engine


def get_vad_engine() -> SileroVadEngine:
    """回傳已載入的 SileroVadEngine。

    Raises:
        RuntimeError: 尚未呼叫 load() 時。
    """
    if _vad_engine is None:
        raise RuntimeError("SileroVadEngine 尚未載入，請先呼叫 model_loader.load()")
    return _vad_engine
