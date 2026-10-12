"""測試用 stub：在匯入 transcription.service 前隔離沉重依賴。"""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

os.environ.setdefault("LOCAL_MODE", "1")

src = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(src))

# GPU 原生套件在此環境無法載入；測試只走 mock engine，不需真實 faster-whisper。
_fw = MagicMock()
_fw.transcribe = MagicMock()
_fw.transcribe.Segment = object
sys.modules.setdefault("faster_whisper", _fw)
sys.modules.setdefault("faster_whisper.transcribe", _fw.transcribe)

for _mod in (
    "ctranslate2",
    "ctranslate2._ext",
    "faster_whisper.audio",
    "faster_whisper.feature_extractor",
    "faster_whisper.utils",
    "faster_whisper.version",
    "torch",
    "torchaudio",
    "silero_vad",
    "onnxruntime",
    "whisperx",
    "whisperx.diarize",
    "transcription.diarization",
    "transcription.engine",
    "transcription.async_faster_whisper",
):
    sys.modules.setdefault(_mod, MagicMock())
