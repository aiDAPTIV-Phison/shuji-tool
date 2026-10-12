from types import SimpleNamespace

from transcription.service import TranscriptionService


def _segment(start: float, end: float, text: str, words=None):
    """建立測試用的 segment 物件。

    Args:
        start: 片段起始秒數。
        end: 片段結束秒數。
        text: 片段文字。
        words: 可選的字級時間戳清單。

    Returns:
        具有 start、end、text、words 屬性的 SimpleNamespace。
    """
    return SimpleNamespace(start=start, end=end, text=text, words=words)


def test_format_transcription_without_word_timestamps_emits_one_line_per_segment():
    service = TranscriptionService.__new__(TranscriptionService)
    segments = [
        _segment(0.0, 2.4, " 大家好"),
        _segment(2.4, 5.1, " 今天天氣不錯"),
    ]

    result = service.format_transcription(
        segments, word_timestamps=False, always_speaker_label=True
    )

    assert result == (
        "[SPEAKER_00]|0.00s -> 2.40s| 大家好\n"
        "[SPEAKER_00]|2.40s -> 5.10s| 今天天氣不錯"
    )


def test_format_transcription_without_word_timestamps_skips_blank_segments():
    service = TranscriptionService.__new__(TranscriptionService)
    segments = [
        _segment(0.0, 1.0, "   "),
        _segment(1.0, 2.0, "有效"),
        _segment(2.0, 3.0, ""),
    ]

    result = service.format_transcription(
        segments, word_timestamps=False, always_speaker_label=True
    )

    assert result == "[SPEAKER_00]|1.00s -> 2.00s| 有效"


def test_format_transcription_without_word_timestamps_empty_input_returns_empty():
    service = TranscriptionService.__new__(TranscriptionService)

    assert service.format_transcription([], word_timestamps=False) == ""


def test_format_transcription_default_concatenates_segment_text():
    service = TranscriptionService.__new__(TranscriptionService)
    segments = [
        _segment(0.0, 2.4, " 大家好"),
        _segment(2.4, 5.1, " 今天天氣不錯"),
    ]

    result = service.format_transcription(segments, word_timestamps=False)

    assert result == "大家好 今天天氣不錯"


def test_format_transcription_word_timestamps_emits_word_lines_without_speaker():
    service = TranscriptionService.__new__(TranscriptionService)
    segments = [
        _segment(
            0.0,
            1.0,
            " 大家好",
            words=[
                SimpleNamespace(start=0.0, end=0.4, word=" 大"),
                SimpleNamespace(start=0.4, end=1.0, word="家好"),
            ],
        ),
    ]

    result = service.format_transcription(segments, word_timestamps=True)

    assert result == (
        "|0.00s -> 0.40s|  大\n"
        "|0.40s -> 1.00s| 家好"
    )


def test_format_transcription_word_timestamps_with_speaker_label():
    service = TranscriptionService.__new__(TranscriptionService)
    segments = [
        _segment(
            0.0,
            1.0,
            " 大家好",
            words=[
                SimpleNamespace(start=0.0, end=0.4, word=" 大"),
                SimpleNamespace(start=0.4, end=1.0, word="家好"),
            ],
        ),
    ]

    result = service.format_transcription(
        segments, word_timestamps=True, always_speaker_label=True
    )

    assert result == (
        "[SPEAKER_00]|0.00s -> 0.40s|  大\n"
        "[SPEAKER_00]|0.40s -> 1.00s| 家好"
    )
