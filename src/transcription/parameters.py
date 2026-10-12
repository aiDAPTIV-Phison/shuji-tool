
from typing import Optional
from pydantic import BaseModel, Field

from utils.ConfigHandler import ConfigHandler

def normalize_language(language: Optional[str]) -> Optional[str]:
    """將呼叫端 Language 正規化為引擎可接受的值。

    None 與空白字串表示自動偵測；有值則去除首尾空白並轉小寫。

    Args:
        language: 呼叫端傳入的語言代碼，可為 None。

    Returns:
        正規化後的語言代碼；自動偵測時回傳 None。
    """
    if language is None:
        return None
    normalized = language.strip().lower()
    if not normalized:
        return None
    return normalized

class WhisperParams(BaseModel):
    """
    WhisperParams 類別，用於定義 Whisper 模型的參數
    預設值由 VTT Gateway 提供
    """
    language: Optional[str]
    beam_size: int
    condition_on_previous_text: bool
    initial_prompt: Optional[str]
    hotwords: Optional[str]
    repetition_penalty: float
    use_fast_pipeline: bool
    enable_vad: bool
    vad_filter_ms: Optional[int]
    vad_threshold: Optional[float]
    neg_threshold: Optional[float]
    min_speech_duration_ms: Optional[int]
    max_speech_duration_s: Optional[float]
    speech_pad_ms: Optional[int]
    word_timestamps: bool
    always_speaker_label: bool = False

class DiarizationParams(BaseModel):
    """
    DiarizationParams 類別，用於定義說話者分離參數。
    """
    diarization: bool
    min_speaker: Optional[int] = None
    max_speaker: Optional[int] = None

def get_default_whisper_params() -> WhisperParams:
    return WhisperParams(
        language="zh",
        beam_size=ConfigHandler.get_whisper_beam_size(),
        use_fast_pipeline=False,
        hotwords=None,
        condition_on_previous_text=True,
        initial_prompt="Ignoring repeated",
        repetition_penalty=1.0,
        # VAD 參數
        enable_vad=True,
        vad_filter_ms=1000,
        vad_threshold=0.5,
        neg_threshold=None,
        min_speech_duration_ms=None,
        max_speech_duration_s=None,
        speech_pad_ms=None,
        word_timestamps=False,
        always_speaker_label=False,
    )