from abc import ABC, abstractmethod
# from faster_whisper import WhisperModel, BatchedInferencePipeline
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment, TranscriptionInfo
from transcription.async_faster_whisper import AsyncFasterWhisperEngine
from transcription.parameters import WhisperParams
from typing import Tuple
import asyncio

from utils.utils import is_cuda_available
from utils.logger import Logger
from utils.call_site import get_call_site_info
from utils.LogHandler import LogHandler as lh


class FasterWhisperEngine(AsyncFasterWhisperEngine):
    """
    Faster Whisper 引擎的具體實現。
    """
    def __init__(self, model_size: str):
        # 初始化 AsyncFasterWhisperEngine
        super().__init__(model_size)
        
        # 保留原有的 logger 和設定
        self.logger = Logger(name="FasterWhisperEngineLogger", log_file="logs/transcription_engine.log")

    async def transcribe_async_method(self, flow_id: str, audio_path: str, params: WhisperParams) -> Tuple[list[Segment], TranscriptionInfo]:
        """
        異步方法，供需要異步調用的地方使用
        """
        callSite = get_call_site_info()
        lh.handle_info(self.logger, flow_id, callSite, f"Starting async transcription. flow_id={flow_id}")

        segments, info = await self.transcribe_async(flow_id=flow_id, audio_path=audio_path, params=params)
        
        lh.handle_info(self.logger, flow_id, callSite, f"Async transcription completed successfully. flow_id={flow_id}, segment_count={len(segments)}")
        
        return segments, info
