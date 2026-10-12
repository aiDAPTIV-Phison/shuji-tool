import pytest

from transcription.parameters import normalize_language


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("", None),
        ("   ", None),
        ("zh", "zh"),
        ("ZH", "zh"),
        (" zh ", "zh"),
        ("En", "en"),
    ],
)
def test_normalize_language(raw, expected):
    """驗證 Language 正規化：None／空白為自動偵測，其餘 strip 後轉小寫。

    Args:
        raw: 呼叫端傳入的原始語言值。
        expected: 預期的正規化結果。
    """
    assert normalize_language(raw) == expected
