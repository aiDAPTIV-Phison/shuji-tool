"""OpenAI 相容轉錄端點的回應模型與序列化。

faster-whisper 的 Segment 欄位與 OpenAI verbose_json 的 segment 幾乎一對一對應，
因此序列化主要是補上 OpenAI 專屬的 task／duration，以及套用 OpenCC 簡繁轉換。

OpenCC 以「逐段轉換後串接」而非「串接後整段轉換」實作，確保回應內
`text` 與 `segments[].text` 的串接結果一致。
"""

from typing import Any, Iterable, Optional

from pydantic import BaseModel

from transcription.service import cc

RESPONSE_FORMAT_JSON = "json"
RESPONSE_FORMAT_VERBOSE_JSON = "verbose_json"
RESPONSE_FORMAT_TEXT = "text"
SUPPORTED_RESPONSE_FORMATS = (
    RESPONSE_FORMAT_JSON,
    RESPONSE_FORMAT_VERBOSE_JSON,
    RESPONSE_FORMAT_TEXT,
)


class VerboseWord(BaseModel):
    """verbose_json 的字級時間戳。"""

    word: str
    start: float
    end: float
    speaker: Optional[str] = None


class VerboseSegment(BaseModel):
    """verbose_json 的片段，欄位對齊 faster-whisper Segment。"""

    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: list[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    speaker: Optional[str] = None

class TranscriptionJsonResponse(BaseModel):
    """response_format=json 的回應。"""

    text: str


class TranscriptionVerboseResponse(BaseModel):
    """response_format=verbose_json 的回應。"""

    task: str = "transcribe"
    language: str
    duration: float
    text: str
    segments: list[VerboseSegment]
    words: Optional[list[VerboseWord]] = None


def build_text(segments: Iterable[Any]) -> str:
    """將片段串接為完整轉錄文字。

    Args:
        segments: 已清理的轉錄片段。

    Returns:
        逐段簡繁轉換後串接的文字。
    """
    return "".join(cc.convert(segment.text) for segment in segments).strip()


def _build_segment(index: int, segment: Any) -> VerboseSegment:
    """將單一片段轉為 verbose_json segment。

    Whisper 專屬的解碼統計欄位以 getattr 取值，讓非 faster-whisper 來源
    （例如遠端引擎組出的片段）也能序列化。

    Args:
        index: 片段序號，作為缺少 id 時的替代值。
        segment: 轉錄片段。

    Returns:
        VerboseSegment。
    """
    return VerboseSegment(
        id=getattr(segment, "id", index),
        seek=getattr(segment, "seek", 0),
        start=segment.start,
        end=segment.end,
        text=cc.convert(segment.text),
        tokens=list(getattr(segment, "tokens", None) or []),
        temperature=getattr(segment, "temperature", 0.0),
        avg_logprob=getattr(segment, "avg_logprob", 0.0),
        compression_ratio=getattr(segment, "compression_ratio", 0.0),
        no_speech_prob=getattr(segment, "no_speech_prob", 0.0),
        speaker=getattr(segment, "speaker", None),
    )


def _build_words(segments: Iterable[Any]) -> list[VerboseWord]:
    """收集所有片段的字級時間戳。

    逐字套用簡繁轉換以保住時間戳對應，代價是轉換品質低於整段轉換，
    因此 words 串接結果可能與 segment text 略有差異。

    Args:
        segments: 轉錄片段。

    Returns:
        跨片段攤平的字級時間戳清單。
    """
    words: list[VerboseWord] = []
    for segment in segments:
        for word in getattr(segment, "words", None) or []:
            words.append(
                VerboseWord(
                    word=cc.convert(word.word),
                    start=word.start,
                    end=word.end,
                    speaker=getattr(word, "speaker", None),
                )
            )
    return words


def build_verbose_response(
    segments: Iterable[Any],
    info: Any,
    include_words: bool,
) -> TranscriptionVerboseResponse:
    """組出 verbose_json 回應。

    duration 取轉錄資訊的原始音檔長度，而非片段時長總和；砍頭砍尾會移除片段，
    用總和會低估時長，導致 LiteLLM 成本計算失準。

    Args:
        segments: 已清理的轉錄片段。
        info: 引擎回傳的轉錄資訊。
        include_words: 是否輸出字級時間戳。

    Returns:
        TranscriptionVerboseResponse。
    """
    segment_list = list(segments)
    verbose_segments = [_build_segment(index, segment) for index, segment in enumerate(segment_list)]
    return TranscriptionVerboseResponse(
        language=getattr(info, "language", None) or "unknown",
        duration=float(getattr(info, "duration", 0.0) or 0.0),
        text="".join(segment.text for segment in verbose_segments).strip(),
        segments=verbose_segments,
        words=_build_words(segment_list) if include_words else None,
    )
