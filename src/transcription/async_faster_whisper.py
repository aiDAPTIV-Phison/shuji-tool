import asyncio
from typing import Tuple
from abc import ABC, abstractmethod

from transcription.parameters import WhisperParams
from transcription.cuda_exception import cuda_exception_classifier, CudaException
from faster_whisper import WhisperModel, BatchedInferencePipeline
from faster_whisper.transcribe import (
    TranscriptionInfo, Segment
)
from utils.logger import Logger
from utils.call_site import get_call_site_info
from utils.LogHandler import LogHandler as lh
from utils.utils import is_cuda_available
from utils.ConfigHandler import ConfigHandler

class AsyncTranscriptionEngine(ABC):
    """
    異步音頻轉錄引擎的抽象基底類別
    """
    @abstractmethod
    async def transcribe_async(self, flow_id: str, audio_path: str, params: WhisperParams) -> Tuple[list[Segment], TranscriptionInfo]:
        pass


class AsyncFasterWhisperEngine(AsyncTranscriptionEngine):
    """
    異步 Faster Whisper 引擎的實現
    """
    def __init__(self, model_size: str):
        self.logger = Logger(name="AsyncFasterWhisperEngineLogger", 
                           log_file="logs/async_transcription_engine.log")
        callSite = get_call_site_info()
        
        try:
            # 確定設備
            device = "cuda" if is_cuda_available() else "cpu"
            compute_type = "float16" if device == 'cuda' else "int8"
            
            # 紀錄設備信息
            lh.handle_info(self.logger, "AsyncFasterWhisperEngine", callSite, 
                         f"Using device: {device}, compute_type: {compute_type}")
            
            # 初始化模型
            model_num_workers = 2
            lh.handle_info(self.logger, "AsyncFasterWhisperEngine", callSite, 
                         f"Initializing Async Faster Whisper model with size: {model_size}")
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type, 
                                    num_workers=model_num_workers, device_index=[0])
            self.fast_pipeline = BatchedInferencePipeline(self.model)
            self.transcribe_semaphore = asyncio.Semaphore(model_num_workers)
            
            # 日誌記錄初始化成功
            lh.handle_info(self.logger, "AsyncFasterWhisperEngine", callSite, 
                         f"Async Model initialized successfully. model_size={model_size}, device={device}, compute_type={compute_type}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Async Faster Whisper model. Error: {str(e)}")
            raise e

    async def transcribe_async(self, flow_id: str, audio_path: str, params: WhisperParams) -> Tuple[list[Segment], TranscriptionInfo]:
        """
        異步轉錄音頻文件
        """
        callSite = get_call_site_info()
        lh.handle_info(self.logger, flow_id, callSite, 
                     f"Starting async transcription. flow_id={flow_id} params={params}")
        
        try:
            async with self.transcribe_semaphore:
                if params.use_fast_pipeline:
                    segments, info = await asyncio.to_thread(
                        self._run_fast_batch,
                        flow_id, audio_path, params
                    )
                else:
                    segments, info = await asyncio.to_thread(
                        self._run_accurate_model,
                        flow_id, audio_path, params
                    )

            lh.handle_info(self.logger, flow_id, callSite, 
                         f"Async transcription completed successfully. flow_id={flow_id}, segment_count={len(segments)}")
            
            return segments, info
            
        except Exception as e:
            classified = cuda_exception_classifier(e)
            if isinstance(classified, CudaException):
                self.logger.error(f"CUDA exception occurred. flow_id={flow_id}, audio_path={audio_path}, error={str(e)}")
                raise classified from e
            
            self.logger.error(f"Failed to async transcribe audio file. flow_id={flow_id}, audio_path={audio_path}, error={str(e)}")
            raise
            
    def _build_vad_parameters(self, params: WhisperParams) -> dict:
        vad_parameters: dict = {}
        if params.vad_filter_ms is not None:
            vad_parameters["min_silence_duration_ms"] = params.vad_filter_ms
        if params.vad_threshold is not None:
            vad_parameters["threshold"] = params.vad_threshold
        if params.neg_threshold is not None:
            vad_parameters["neg_threshold"] = params.neg_threshold
        if params.min_speech_duration_ms is not None:
            vad_parameters["min_speech_duration_ms"] = params.min_speech_duration_ms
        if params.max_speech_duration_s is not None:
            vad_parameters["max_speech_duration_s"] = params.max_speech_duration_s
        if params.speech_pad_ms is not None:
            vad_parameters["speech_pad_ms"] = params.speech_pad_ms
        return vad_parameters

    def _run_fast_batch(self, flow_id: str, audio_path: str, params: WhisperParams) -> Tuple[list[Segment], TranscriptionInfo]:
        """
        使用Pipeline加速方法，必須開VAD過濾
        """
        lh.handle_info(self.logger, flow_id, get_call_site_info(), 
                f"Use fast pipeline. flow_id={flow_id}")
        batch_size = ConfigHandler.get_batch_size()
        segments, info = self.fast_pipeline.transcribe(
            str(audio_path),
            batch_size=batch_size,
            beam_size = params.beam_size,
            language = params.language,
            hotwords = params.hotwords,
            condition_on_previous_text=params.condition_on_previous_text,
            initial_prompt=params.initial_prompt,
            repetition_penalty=params.repetition_penalty,            
            vad_filter=True,
            vad_parameters=self._build_vad_parameters(params),
            word_timestamps=params.word_timestamps,
        )

        return list(segments), info
            
    def _run_accurate_model(self, flow_id: str, audio_path: str, params: WhisperParams) -> Tuple[list[Segment], TranscriptionInfo]:
        """
        使用模型轉錄方法，準確率比Pipeline高
        """
        lh.handle_info(self.logger, flow_id, get_call_site_info(), 
                f"Use accurate model. flow_id={flow_id}")
        segments, info = self.model.transcribe(
            str(audio_path),
            beam_size = params.beam_size,
            language = params.language,
            hotwords = params.hotwords,
            condition_on_previous_text=params.condition_on_previous_text,
            initial_prompt=params.initial_prompt,
            repetition_penalty=params.repetition_penalty,
            vad_filter=params.enable_vad,
            vad_parameters=self._build_vad_parameters(params),
            word_timestamps=params.word_timestamps,
        )

        return list(segments), info
