"""DiarizationEngine 的 Speaker block 合併與字串格式化迴歸。"""

import importlib.util
from pathlib import Path

import pytest


def _load_diarization_engine_class():
    """以獨立模組名載入真實 diarization.py（略過 conftest 的 MagicMock）。

    Returns:
        DiarizationEngine 類別。
    """
    path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "transcription"
        / "diarization.py"
    )
    spec = importlib.util.spec_from_file_location("diarization_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.DiarizationEngine


@pytest.fixture(scope="module")
def diarization_engine_cls():
    """提供可呼叫 merge／format 的 DiarizationEngine 類別。"""
    return _load_diarization_engine_class()


@pytest.fixture
def engine(diarization_engine_cls):
    """未初始化的引擎實例，僅供純函式方法測試。"""
    return diarization_engine_cls.__new__(diarization_engine_cls)


def test_merge_speaker_blocks_merges_consecutive_same_speaker(engine):
    """連續同語者合併為一塊，換語者則切開。"""
    segments = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0, "text": "大家好"},
        {"speaker": "SPEAKER_00", "start": 1.0, "end": 2.0, "text": "我是"},
        {"speaker": "SPEAKER_01", "start": 2.0, "end": 3.5, "text": "小明"},
    ]
    blocks = engine.merge_speaker_blocks(segments)
    assert blocks == [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0, "text": "大家好 我是"},
        {"speaker": "SPEAKER_01", "start": 2.0, "end": 3.5, "text": "小明"},
    ]


def test_merge_speaker_blocks_treats_missing_speaker_as_unknown(engine):
    """缺少 speaker 時視為 UNKNOWN。"""
    blocks = engine.merge_speaker_blocks(
        [{"start": 0.0, "end": 1.0, "text": "無人稱"}]
    )
    assert blocks == [
        {"speaker": "UNKNOWN", "start": 0.0, "end": 1.0, "text": "無人稱"}
    ]


def test_format_transcription_uses_speaker_blocks(engine):
    """非字級輸出與合併後的 Gateway pipe 字串一致。"""
    segments = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0, "text": "大家好"},
        {"speaker": "SPEAKER_00", "start": 1.0, "end": 2.0, "text": "我是"},
        {"speaker": "SPEAKER_01", "start": 2.0, "end": 3.5, "text": "小明"},
    ]
    result = engine.format_transcription(segments, word_timestamps=False)
    assert result == (
        "[SPEAKER_00]|0.00s -> 2.00s| 大家好 我是\n"
        "[SPEAKER_01]|2.00s -> 3.50s| 小明"
    )


def test_format_transcription_word_timestamps_emits_per_word_lines(engine):
    """字級輸出一字一行，不合塊。"""
    segments = [
        {
            "speaker": "SPEAKER_00",
            "start": 0.0,
            "end": 1.0,
            "text": "大家好",
            "words": [
                {"word": "大", "start": 0.0, "end": 0.4, "speaker": "SPEAKER_00"},
                {"word": "家好", "start": 0.4, "end": 1.0, "speaker": "SPEAKER_00"},
            ],
        }
    ]
    result = engine.format_transcription(segments, word_timestamps=True)
    assert result == (
        "[SPEAKER_00]|0.00s -> 0.40s| 大\n"
        "[SPEAKER_00]|0.40s -> 1.00s| 家好"
    )
