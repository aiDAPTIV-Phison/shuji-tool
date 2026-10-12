from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

from opencc import OpenCC
from faster_whisper.transcribe import Segment, TranscriptionInfo

from transcription.engine import AsyncFasterWhisperEngine, FasterWhisperEngine
from transcription.parameters import DiarizationParams, WhisperParams
from utils.logger import Logger
from utils.call_site import get_call_site_info
from utils.LogHandler import LogHandler as lh
from utils.ConfigHandler import ConfigHandler

if TYPE_CHECKING:
    from transcription.diarization import DiarizationEngine

cc = OpenCC('s2tw')

REMOVE_FIRST_SEGMENT_NO_SPEECH_PROB = ConfigHandler.get_remove_first_segment_no_speech_prob()
REMOVE_FIRST_SEGMENT_LOG_PROB = ConfigHandler.get_remove_first_segment_log_prob()
REMOVE_LAST_SEGMENT_DURATION = ConfigHandler.get_remove_last_segment_duration()
REMOVE_LAST_SEGMENT_NO_SPEECH_PROB = ConfigHandler.get_remove_last_segment_no_speech_prob()
HARD_NO_SPEECH_PROB_THRESHOLD = ConfigHandler.get_hard_no_speech_prob_threshold()

HALLUCINATION_BLACKLIST = [
    "以上言論不代表本台立場",
    "字幕由 Amara.org 社群提供",
    "字幕由Amara.org社群提供",
    "歡迎訂閱",
    "按讚訂閱分享",
    "點讚訂閱轉發",
    "MING PAO CANADA",
    "MING PAO TORONTO",
    "明鏡與點點欄目",
]
HALLUCINATION_BLACKLIST = sorted(HALLUCINATION_BLACKLIST, key=len, reverse=True)


@dataclass
class ChunkDecodeResult:
    """無狀態 chunk 解碼結果；diarization 成功時才帶 turns／speakers。"""

    transcription: str
    turns: list[dict[str, Any]] | None = None
    speakers: list[dict[str, Any]] | None = None


def segment_info_to_string(segment: Segment) -> str:
    return (
        f"no_speech_prob={segment.no_speech_prob:.5f}, prob={segment.avg_logprob:.5f}, " 
        f"duration={segment.end - segment.start}, compression_ratio={segment.compression_ratio:.5f}, temperature={segment.temperature:.5f}"
    )

class TranscriptionService:
    """
    高層的服務邏輯管控層，封裝業務流程。
    """
    def __init__(self, engine: FasterWhisperEngine, diarization_engine: DiarizationEngine | None = None):
        callSite = get_call_site_info()
        # 初始化 Logger
        self.logger = Logger(name="TranscriptionServiceLogger", log_file="logs/transcription_service.log")
        self.engine = engine
        self.diarization_engine = diarization_engine
        lh.handle_info(self.logger, "TranscriptionService", callSite, f"TranscriptionService initialized with TranscriptionEngine.")

    async def transcribe_to_segments(
        self,
        flow_id: str,
        audio_path: str,
        params: WhisperParams,
    ) -> tuple[list[Segment], TranscriptionInfo]:
        """轉錄音檔並清理幻覺片段，回傳結構化片段。

        供需要 segment 級資訊的呼叫端使用（例如 OpenAI 相容端點的 verbose_json）；
        不做文字格式化，也不做簡繁轉換。

        Args:
            flow_id (str): 流程唯一識別符
            audio_path (str): 音頻檔案的路徑
            params (WhisperParams): 轉錄參數

        Returns:
            tuple[list[Segment], TranscriptionInfo]: 已清理的轉錄片段與轉錄資訊
        """
        callSite = get_call_site_info()
        segments, info = await self.engine.transcribe_async_method(
            flow_id=flow_id, audio_path=audio_path, params=params
        )
        lh.handle_info(
            self.logger,
            flow_id,
            callSite,
            f"Detected language. flow_id={flow_id}, requested_language={params.language}, "
            f"detected_language={info.language}, language_probability={info.language_probability}",
        )
        return self._cleanse_segments(flow_id, segments), info

    async def process_audio(
            self,
            flow_id: str,
            audio_path: str,
            params: WhisperParams,
            diarization_params: DiarizationParams | None = None,
        ) -> str:
        """
        使用選定的轉錄引擎處理音頻並進行後處理

        Args:
            flow_id (str): 流程唯一識別符
            audio_path (str): 音頻檔案的路徑
            params (WhisperParams): 轉錄參數
            diarization_params (DiarizationParams | None): 說話者分離參數
        Returns:
            str: 轉錄結果
        """
        callSite = get_call_site_info()
        lh.handle_info(self.logger, flow_id, callSite, f"Processing audio file. flow_id={flow_id}, audio_path={audio_path}")
        
        post_processed_segments, info = await self.transcribe_to_segments(
            flow_id=flow_id, audio_path=audio_path, params=params
        )
        if diarization_params and diarization_params.diarization and self.diarization_engine and self.diarization_engine.is_enable:
            result = self.diarization_engine.diarize(flow_id, post_processed_segments, info, audio_path, diarization_params, params.word_timestamps)
        else:
            result = self.format_transcription(
                post_processed_segments,
                params.word_timestamps,
                always_speaker_label=params.always_speaker_label,
            )
        result = cc.convert(result)
        lh.handle_info(self.logger, flow_id, callSite, f"Transcription completed for flow_id={flow_id}. result_length={len(result)}")
        return result

    def format_transcription(
        self,
        segments: Iterable[Segment],
        word_timestamps: bool,
        always_speaker_label: bool = False,
    ) -> str:
        """
        將轉錄片段合併成文字
        """
        # 3.0.0 預設：不加 speaker / 時間戳，直接串接 segment 文字
        if not word_timestamps and not always_speaker_label:
            return "".join(segment.text for segment in segments).strip()

        speaker_prefix = "[SPEAKER_00]" if always_speaker_label else ""
        lines = []
        if not word_timestamps:
            for segment in segments:
                text = segment.text.strip()
                if not text:
                    continue
                lines.append(
                    f"{speaker_prefix}|{segment.start:.2f}s -> {segment.end:.2f}s| {text}\n"
                )
            return "".join(lines).strip()

        # word_timestamps：輸出字級時間列；always_speaker_label 時加上 SPEAKER_00
        for segment in segments:
            if not segment.text.strip():
                continue
            if not segment.words:
                continue
            for word in segment.words:
                lines.append(f"{speaker_prefix}|{word.start:.2f}s -> {word.end:.2f}s| {word.word}\n")

        return "".join(lines).strip()

    async def decode_chunk(self, flow_id: str, audio_path: str, params: WhisperParams, diarization: bool = False) -> ChunkDecodeResult:
        """對單一音訊 chunk 做無狀態解碼，不做批次檔案的砍頭砍尾。

        Args:
            flow_id: 流程唯一識別符。
            audio_path: 音訊檔路徑。
            params: 轉錄參數。

        Returns:
            簡轉繁與 blacklist 過濾後的解碼文字。
        """
        callSite = get_call_site_info()
        lh.handle_debug(
            self.logger,
            flow_id,
            callSite,
            f"Decoding audio chunk. flow_id={flow_id}, audio_path={audio_path}",
        )

        segments, info = await self.engine.transcribe_async_method(
            flow_id=flow_id, audio_path=audio_path, params=params
        )
        lh.handle_debug(
            self.logger,
            flow_id,
            callSite,
            f"Detected language. flow_id={flow_id}, requested_language={params.language}, "
            f"detected_language={info.language}, language_probability={info.language_probability}",
        )

        kept = []
        for segment in segments:
            matched_blacklist = self.should_remove_hallucination_segment(segment.text)
            if matched_blacklist:
                self._log_removed_segment(flow_id, "blacklist_match", segment, matched_blacklist)
                continue
            if not segment.text.strip():
                continue
            kept.append(segment)

        result = "".join(segment.text for segment in kept).strip()
        result = cc.convert(result)
        lh.handle_debug(
            self.logger,
            flow_id,
            callSite,
            f"Chunk decode completed for flow_id={flow_id}. result_length={len(result)}",
        )
        turns = None
        speakers = None
        if diarization:
            turns, speakers = self._try_diarize_chunk(flow_id, kept, info, audio_path)
        return ChunkDecodeResult(transcription=result, turns=turns, speakers=speakers)

    def _cleanse_segments(self, flow_id: str, segments: Iterable[Segment]) -> list[Segment]:
        """
        清理有幻覺的轉錄片段
        """
        segment_list = list(segments)
        if not segment_list:
            return []

        # 移除頭尾幻覺片段
        segment_list = self._remove_first_segment_hallucination(flow_id, segment_list)
        segment_list = self._remove_last_segment_hallucination(flow_id, segment_list)

        post_processed_segments = []
        for segment in segment_list:
            self.logger.debug(
                f"Transcription segment: flow_id={flow_id}, text={segment.text}, "
                f"{segment_info_to_string(segment)}"
            )
            matched_blacklist = self.should_remove_hallucination_segment(segment.text)
            if matched_blacklist:
                self._log_removed_segment(flow_id, "blacklist_match", segment, matched_blacklist)
                continue
            if not segment.text.strip():
                continue
            post_processed_segments.append(segment)

        return post_processed_segments
    
    def _remove_first_segment_hallucination(self, flow_id: str, segments: list[Segment]) -> list[Segment]:
        """
        移除開頭幻覺片段
        """
        segment_list = list(segments)
        if not segment_list:
            return segment_list

        first_segment = segment_list[0]
        self.logger.debug(
            f"First Segement Info. flow_id={flow_id}, first_segment={first_segment.text}, "
            f"{segment_info_to_string(first_segment)}"
        )
        if first_segment.no_speech_prob >= HARD_NO_SPEECH_PROB_THRESHOLD:
            segment_list.pop(0)
            self._log_removed_segment(flow_id, "first_hard_no_speech", first_segment)
        elif first_segment.avg_logprob <= REMOVE_FIRST_SEGMENT_LOG_PROB and first_segment.no_speech_prob >= REMOVE_FIRST_SEGMENT_NO_SPEECH_PROB:
            segment_list.pop(0)
            self._log_removed_segment(flow_id, "first_soft_threshold", first_segment)

        return segment_list
    
    def _remove_last_segment_hallucination(self, flow_id: str, segments: list[Segment]) -> list[Segment]:
        """
        移除尾端幻覺片段
        """
        segment_list = list(segments)
        if not segment_list:
            return segment_list

        last_segment = segment_list[-1]
        last_segment_duration = last_segment.end - last_segment.start
        self.logger.debug(
            f"Last Segement Info. flow_id={flow_id}, last_segment={last_segment.text}, "
            f"{segment_info_to_string(last_segment)}"
        )
        if last_segment.no_speech_prob >= HARD_NO_SPEECH_PROB_THRESHOLD:
            segment_list.pop()
            self._log_removed_segment(flow_id, "last_hard_no_speech", last_segment)
        elif last_segment_duration <= REMOVE_LAST_SEGMENT_DURATION and last_segment.no_speech_prob >= REMOVE_LAST_SEGMENT_NO_SPEECH_PROB:
            segment_list.pop()
            self._log_removed_segment(flow_id, "last_soft_threshold", last_segment)
        return segment_list

    def _log_removed_segment(
        self,
        flow_id: str,
        reason: str,
        segment: Segment,
        matched_blacklist: str | None = None,
    ) -> None:
        """記錄被移除的轉錄片段完整資訊。

        Args:
            flow_id: 流程唯一識別符。
            reason: 觸發移除的規則名稱。
            segment: 被移除的轉錄片段。
            matched_blacklist: 命中的 blacklist 字串，非 blacklist 移除時可省略。
        """
        message = (
            f"Remove segment. flow_id={flow_id}, reason={reason}, text={segment.text}, "
            f"{segment_info_to_string(segment)}"
        )
        if matched_blacklist is not None:
            message += f", matched_blacklist={matched_blacklist}"
        self.logger.info(message)
    
    def should_remove_hallucination_segment(self, text: str) -> str | None:
        """判斷轉錄片段是否包含已知幻覺文字，若包含則移除整個 segment。"""
        for hallucination_text in HALLUCINATION_BLACKLIST:
            if hallucination_text in text:
                return hallucination_text
        return None

    def _try_diarize_chunk(
        self,
        flow_id: str,
        segments: list[Segment],
        info: Any,
        audio_path: str,
    ) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]] | None]:
        """嘗試對本包音做講者分離；失敗時回 (None, None) 以便只保留稿。

        Args:
            flow_id: 流程唯一識別符。
            segments: 已過濾的 ASR 片段。
            info: faster-whisper 轉錄資訊（含偵測語言）。
            audio_path: 本包 wav 路徑。

        Returns:
            (turns, speakers)；引擎未載入或失敗時為 (None, None)。
        """
        engine = self.diarization_engine
        if engine is None or not getattr(engine, "is_enable", False):
            return None, None
        try:
            chunk = engine.diarize_chunk(flow_id, segments, info, audio_path)
            turns = [
                {"text": cc.convert(turn.text), "local_speaker": turn.local_speaker}
                for turn in chunk.turns
            ]
            speakers = [
                {
                    "local_speaker": speaker.local_speaker,
                    "embeddings": list(speaker.embeddings),
                }
                for speaker in chunk.speakers
            ]
            return turns, speakers
        except Exception as e:
            lh.handle_warning(
                self.logger,
                flow_id,
                get_call_site_info(),
                f"Chunk diarization failed; returning transcription only. flow_id={flow_id}, error={str(e)}",
            )
            return None, None
