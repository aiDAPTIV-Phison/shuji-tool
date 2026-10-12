from typing import Optional

from pydantic import BaseModel


class VadParams(BaseModel):
    """即時 VAD 選填偵測旋鈕；未傳的欄位保持 None，交給 silero-vad 套件預設。"""

    vad_threshold: Optional[float] = None
    vad_filter_ms: Optional[int] = None
    neg_threshold: Optional[float] = None
    min_speech_duration_ms: Optional[int] = None
    max_speech_duration_s: Optional[float] = None
    speech_pad_ms: Optional[int] = None
