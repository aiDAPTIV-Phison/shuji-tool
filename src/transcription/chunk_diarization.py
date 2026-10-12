from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass
class ChunkTurn:
    """本包音內一個講者連續片段。"""

    text: str
    local_speaker: str


@dataclass
class ChunkSpeaker:
    """本包音內一個 local_speaker 與其聲紋向量。"""

    local_speaker: str
    embeddings: list[float]


@dataclass
class ChunkDiarization:
    """WhisperX 切字後的 turns 與每位講者一條 embedding。"""

    turns: list[ChunkTurn]
    speakers: list[ChunkSpeaker]


def format_chunk_diarization(
    segments: Iterable[dict[str, Any]],
    embeddings: Mapping[str, list[float]] | None,
) -> ChunkDiarization:
    """將 WhisperX assign_word_speakers 結果編成 turns 與 speakers。

    相鄰且同一 local_speaker 的字／片段會合併；插話不可跨過中間講者合併。
    只為曾出現在 turns 的講者各列一條非空 float 向量。

    Args:
        segments: WhisperX segments（可含 words 與 speaker）。
        embeddings: local_speaker 到一條 embedding 的對應；缺少或非法時拋錯。

    Returns:
        時間序 turns，以及 turns 中每位講者恰好一列 speakers。

    Raises:
        ValueError: embeddings 缺失、對不上 turns，或不是非空 float 陣列。
    """
    pieces: list[tuple[str, str]] = []
    for segment in segments:
        words = segment.get("words") or []
        labeled_words = [
            word
            for word in words
            if (word.get("word") or "").strip()
            and (word.get("speaker") or segment.get("speaker"))
        ]
        if labeled_words:
            for word in labeled_words:
                speaker = word.get("speaker") or segment.get("speaker")
                pieces.append((speaker, word.get("word") or ""))
            continue
        speaker = segment.get("speaker")
        text = segment.get("text") or ""
        if speaker and text.strip():
            pieces.append((speaker, text))

    turns: list[ChunkTurn] = []
    for speaker, text in pieces:
        if turns and turns[-1].local_speaker == speaker:
            turns[-1] = ChunkTurn(
                text=turns[-1].text + text,
                local_speaker=speaker,
            )
        else:
            turns.append(ChunkTurn(text=text, local_speaker=speaker))

    normalized_turns: list[ChunkTurn] = []
    for turn in turns:
        stripped = turn.text.strip()
        if stripped:
            normalized_turns.append(
                ChunkTurn(text=stripped, local_speaker=turn.local_speaker)
            )

    speakers: list[ChunkSpeaker] = []
    seen: set[str] = set()
    for turn in normalized_turns:
        if turn.local_speaker in seen:
            continue
        seen.add(turn.local_speaker)
        vector = _require_embedding(embeddings, turn.local_speaker)
        speakers.append(ChunkSpeaker(local_speaker=turn.local_speaker, embeddings=vector))

    return ChunkDiarization(turns=normalized_turns, speakers=speakers)


def _require_embedding(
    embeddings: Mapping[str, list[float]] | None,
    local_speaker: str,
) -> list[float]:
    """取出並驗證該講者的一條非空 float 向量。

    Args:
        embeddings: local_speaker 到 embedding 的對應。
        local_speaker: 本包音講者標籤。

    Returns:
        複製後的 float 向量。

    Raises:
        ValueError: 對應不存在或不是非空數值陣列。
    """
    if embeddings is None or local_speaker not in embeddings:
        raise ValueError(f"Missing embedding for {local_speaker}")
    raw = embeddings[local_speaker]
    try:
        items = list(raw)
    except TypeError as exc:
        raise ValueError(f"Embedding for {local_speaker} is not a non-empty float array") from exc
    if not items:
        raise ValueError(f"Embedding for {local_speaker} is not a non-empty float array")
    vector: list[float] = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"Embedding for {local_speaker} is not a non-empty float array")
        vector.append(float(item))
    return vector
