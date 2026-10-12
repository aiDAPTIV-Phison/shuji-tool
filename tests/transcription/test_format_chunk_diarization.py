from transcription.chunk_diarization import format_chunk_diarization
import pytest


def test_format_chunk_diarization_merges_adjacent_same_speaker():
    """相鄰且同一 local_speaker 的片段合併成一則 turn。"""
    segments = [
        {
            "text": "你好",
            "speaker": "SPEAKER_00",
            "words": [{"word": "你好", "speaker": "SPEAKER_00"}],
        },
        {
            "text": "大家好",
            "speaker": "SPEAKER_00",
            "words": [{"word": "大家好", "speaker": "SPEAKER_00"}],
        },
        {
            "text": "換我了",
            "speaker": "SPEAKER_01",
            "words": [{"word": "換我了", "speaker": "SPEAKER_01"}],
        },
    ]
    embeddings = {
        "SPEAKER_00": [0.12, -0.04],
        "SPEAKER_01": [0.08, 0.33],
    }

    result = format_chunk_diarization(segments, embeddings)

    assert [(turn.text, turn.local_speaker) for turn in result.turns] == [
        ("你好大家好", "SPEAKER_00"),
        ("換我了", "SPEAKER_01"),
    ]
    assert [(speaker.local_speaker, speaker.embeddings) for speaker in result.speakers] == [
        ("SPEAKER_00", [0.12, -0.04]),
        ("SPEAKER_01", [0.08, 0.33]),
    ]


def test_format_chunk_diarization_does_not_merge_across_other_speaker():
    """插話 A／B／A 不可跨過 B 合併。"""
    segments = [
        {
            "text": "甲",
            "speaker": "SPEAKER_00",
            "words": [{"word": "甲", "speaker": "SPEAKER_00"}],
        },
        {
            "text": "乙",
            "speaker": "SPEAKER_01",
            "words": [{"word": "乙", "speaker": "SPEAKER_01"}],
        },
        {
            "text": "甲又說",
            "speaker": "SPEAKER_00",
            "words": [{"word": "甲又說", "speaker": "SPEAKER_00"}],
        },
    ]
    embeddings = {
        "SPEAKER_00": [0.1],
        "SPEAKER_01": [0.2],
    }

    result = format_chunk_diarization(segments, embeddings)

    assert [(turn.text, turn.local_speaker) for turn in result.turns] == [
        ("甲", "SPEAKER_00"),
        ("乙", "SPEAKER_01"),
        ("甲又說", "SPEAKER_00"),
    ]
    assert [speaker.local_speaker for speaker in result.speakers] == [
        "SPEAKER_00",
        "SPEAKER_01",
    ]


def test_format_chunk_diarization_empty_segments_returns_empty_arrays():
    """沒偵測到講者時回空陣列，不自造 SPEAKER_00。"""
    result = format_chunk_diarization([], {"SPEAKER_00": [0.1]})

    assert result.turns == []
    assert result.speakers == []


def test_format_chunk_diarization_drops_embeddings_not_in_turns():
    """speakers 只含 turns 出現過的 local_speaker，恰好一列。"""
    segments = [
        {
            "text": "只有甲",
            "speaker": "SPEAKER_00",
            "words": [{"word": "只有甲", "speaker": "SPEAKER_00"}],
        },
    ]
    embeddings = {
        "SPEAKER_00": [0.1, 0.2],
        "SPEAKER_01": [0.9, 0.8],
    }

    result = format_chunk_diarization(segments, embeddings)

    assert [speaker.local_speaker for speaker in result.speakers] == ["SPEAKER_00"]
    assert result.speakers[0].embeddings == [0.1, 0.2]


def test_format_chunk_diarization_splits_when_words_change_speaker():
    """同一 segment 內詞級講者切換會拆成多則 turn。"""
    segments = [
        {
            "text": "大家好接下來換我說",
            "speaker": "SPEAKER_00",
            "words": [
                {"word": "大家好", "speaker": "SPEAKER_00"},
                {"word": "接下來換我說", "speaker": "SPEAKER_01"},
            ],
        },
    ]
    embeddings = {
        "SPEAKER_00": [0.12, -0.04],
        "SPEAKER_01": [0.08, 0.33],
    }

    result = format_chunk_diarization(segments, embeddings)

    assert [(turn.text, turn.local_speaker) for turn in result.turns] == [
        ("大家好", "SPEAKER_00"),
        ("接下來換我說", "SPEAKER_01"),
    ]


def test_format_chunk_diarization_output_has_no_word_timestamps():
    """turns／speakers 不含時間或 word 陣列。"""
    segments = [
        {
            "text": "大家好",
            "start": 0.0,
            "end": 1.2,
            "speaker": "SPEAKER_00",
            "words": [
                {"word": "大家好", "start": 0.0, "end": 1.2, "speaker": "SPEAKER_00"},
            ],
        },
    ]
    result = format_chunk_diarization(segments, {"SPEAKER_00": [0.5]})

    assert result.turns[0].__dict__ == {"text": "大家好", "local_speaker": "SPEAKER_00"}
    assert result.speakers[0].__dict__ == {
        "local_speaker": "SPEAKER_00",
        "embeddings": [0.5],
    }


def test_format_chunk_diarization_missing_embedding_raises():
    """turns 有講者但 embeddings 對不上時拋錯，讓呼叫端降級。"""
    segments = [
        {
            "text": "大家好",
            "speaker": "SPEAKER_00",
            "words": [{"word": "大家好", "speaker": "SPEAKER_00"}],
        },
    ]

    with pytest.raises(ValueError):
        format_chunk_diarization(segments, {})


def test_format_chunk_diarization_empty_embedding_raises():
    """embeddings 為空陣列時拋錯。"""
    segments = [
        {
            "text": "大家好",
            "speaker": "SPEAKER_00",
            "words": [{"word": "大家好", "speaker": "SPEAKER_00"}],
        },
    ]

    with pytest.raises(ValueError):
        format_chunk_diarization(segments, {"SPEAKER_00": []})


def test_format_chunk_diarization_uses_segment_text_when_words_missing():
    """沒有 words 時用 segment 的 speaker 與 text。"""
    segments = [
        {"text": " 第一句", "speaker": "SPEAKER_00"},
        {"text": " 第二句", "speaker": "SPEAKER_00"},
    ]

    result = format_chunk_diarization(segments, {"SPEAKER_00": [0.1]})

    assert [(turn.text, turn.local_speaker) for turn in result.turns] == [
        ("第一句 第二句", "SPEAKER_00"),
    ]
