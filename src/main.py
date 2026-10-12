"""命令列範例：不經 API 直接呼叫轉錄引擎。

用法:
    python src/main.py <audio_path> [--model <whisper_model_dir>]

模型路徑未指定時取環境變數 WHISPER_MODEL_FOLDER（見 ConfigHandler 預設值）。
"""

import argparse
import asyncio

from transcription.engine import FasterWhisperEngine
from transcription.parameters import get_default_whisper_params
from transcription.service import TranscriptionService
from utils.ConfigHandler import ConfigHandler


def parse_args() -> argparse.Namespace:
    """解析命令列參數。

    Returns:
        含 audio_path 與 model 的 Namespace。
    """
    parser = argparse.ArgumentParser(description="Transcribe one audio file with faster-whisper.")
    parser.add_argument("audio_path", help="待轉錄的音檔路徑")
    parser.add_argument(
        "--model",
        default=ConfigHandler.get_whisper_model_folder(),
        help="CTranslate2 格式的 Whisper 模型目錄；預設取 WHISPER_MODEL_FOLDER",
    )
    return parser.parse_args()


async def transcribe(audio_path: str, model_dir: str) -> str:
    """載入引擎並轉錄單一音檔。

    Args:
        audio_path: 音檔路徑。
        model_dir: Whisper 模型目錄。

    Returns:
        轉錄後的逐字稿文字。
    """
    engine = FasterWhisperEngine(model_size=model_dir)
    service = TranscriptionService(engine=engine)
    return await service.process_audio(
        flow_id="cli",
        audio_path=audio_path,
        params=get_default_whisper_params(),
    )


def main() -> None:
    """命令列入口。"""
    args = parse_args()
    transcript = asyncio.run(transcribe(args.audio_path, args.model))
    print(transcript)


if __name__ == "__main__":
    main()
