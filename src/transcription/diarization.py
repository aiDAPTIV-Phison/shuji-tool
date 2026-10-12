
from typing import Any, Iterable
from dataclasses import dataclass
import gc

import torch
import whisperx
from faster_whisper.transcribe import Segment, TranscriptionInfo
from whisperx.diarize import DiarizationPipeline

from transcription.chunk_diarization import ChunkDiarization, format_chunk_diarization
from transcription.cuda_exception import cuda_exception_classifier, CudaException
from transcription.parameters import DiarizationParams
from utils.utils import is_cuda_available
from utils.logger import Logger
from utils.LogHandler import LogHandler as lh
from utils.call_site import get_call_site_info

@dataclass
class AlignmentProfile:
    model_dir: str
    support_languages: list[str] | str # Support language for alignment.
    load_language_code: str | None = None # Actual language code to load the model in whisperx.load_align_model().

    def get_load_language_code(self) -> str:
        first_support_language = self.support_languages if isinstance(self.support_languages, str) else self.support_languages[0]
        return self.load_language_code or first_support_language

    def get_support_languages(self) -> list[str]:
        if isinstance(self.support_languages, list):
            return self.support_languages
        return [self.support_languages]

@dataclass
class AlignmentInstance:
    profile: AlignmentProfile
    model: Any | None = None
    metadata: dict[str, Any] | None = None

class DiarizationEngine:
    """
    使用 WhisperX 將 faster-whisper 轉錄片段標記說話者並格式化輸出。
    """
    def __init__(
        self,
        device: str | None = None,
        diarization_model_dir: str | None = None,
        alignment_profiles: list[AlignmentProfile] = None,
    ):
        """
        初始化 WhisperX alignment model 與 pyannote diarization pipeline。
        """

        self.is_enable: bool = False
        self.logger = Logger(name="DiarizationEngineLogger", log_file="logs/diarization_engine.log")
        if diarization_model_dir is None or alignment_profiles is None or len(alignment_profiles) == 0:
            self.is_enable = False
            return
        try:
            self.device = device or ("cuda" if is_cuda_available() else "cpu")
            self.alignment_models: dict[str, AlignmentInstance] = {}
            self._create_profile_cache(alignment_profiles)
            # Load the first profile as default
            default_language_code = alignment_profiles[0].get_support_languages()[0]
            self._load_alignment_model(flow_id="INIT", language_code=default_language_code) 

            self.diarization_model_dir = diarization_model_dir
            self.diarization_pipeline = DiarizationPipeline(
                model_name=diarization_model_dir,
                device=self.device,
            )
            self.is_enable = True
        except Exception as e:
            self.is_enable = False
            classified = cuda_exception_classifier(e)
            if isinstance(classified, CudaException):
                self.logger.error(f"CUDA exception occurred during initializing DiarizationEngine. error={str(e)}")
                raise classified from e
            self.logger.error(f"Failed to initialize DiarizationEngine. error={str(e)}")
            raise e

    def diarize(self, flow_id:str, segments: Iterable[Segment], info: TranscriptionInfo, audio_path: str, diarization_params: DiarizationParams, word_timestamps: bool = False) -> str:
        """
        對轉錄片段進行說話者分離，回傳保留 WhisperX speaker 編號的完整文字。
        """
        speaker_segments = self.diarize_to_segments(
            flow_id, segments, info, audio_path, diarization_params
        )
        return self.format_transcription(speaker_segments, word_timestamps)

    def diarize_to_segments(
        self,
        flow_id: str,
        segments: Iterable[Segment],
        info: TranscriptionInfo,
        audio_path: str,
        diarization_params: DiarizationParams,
    ) -> list[dict[str, Any]]:
        """對轉錄片段做語者分離，回傳含 Speaker label 的 WhisperX 片段。

        Args:
            flow_id: 流程唯一識別符。
            segments: faster-whisper 轉錄片段。
            info: 轉錄資訊（含偵測語言）。
            audio_path: 音檔路徑。
            diarization_params: 語者分離參數。

        Returns:
            WhisperX 標記後的片段清單；無輸入片段時為空清單。
        """
        input_segments = self._build_transcript_segments(segments)
        if not input_segments:
            return []

        aligned_result = None
        diarize_segments = None
        speaker_result = None
        try:
            lh.handle_info(self.logger, flow_id, get_call_site_info(), f"Starting Alignment. flow_id={flow_id}, audio_path={audio_path}")
            alignment_model, alignment_metadata = self._load_alignment_model(flow_id=flow_id, language_code=info.language)
            aligned_result = whisperx.align(
                input_segments,
                alignment_model,
                alignment_metadata,
                audio_path,
                self.device,
            )

            lh.handle_info(self.logger, flow_id, get_call_site_info(), f"Starting Diarization. flow_id={flow_id}, audio_path={audio_path}")
            diarization_options = self._build_diarization_options(diarization_params)
            diarize_segments = self.diarization_pipeline(audio_path, **diarization_options)
            speaker_result = whisperx.assign_word_speakers(
                diarize_segments,
                aligned_result,
                fill_nearest=True,
            )
            return list(speaker_result.get("segments", []))
        except Exception as e:
            classified = cuda_exception_classifier(e)
            if isinstance(classified, CudaException):
                self.logger.error(f"CUDA exception occurred during diarization. flow_id={flow_id}, audio_path={audio_path}, error={str(e)}")
                raise classified from e

            self.logger.error(f"Failed to diarize audio file. flow_id={flow_id}, audio_path={audio_path}, error={str(e)}")
            raise
        finally:
            # 先丟掉推論中間結果的引用，再清 CUDA cache，避免 VRAM 只增不減
            del aligned_result, diarize_segments, speaker_result
            self._clear_cuda_cache()

    def merge_speaker_blocks(
        self, segments: Iterable[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """將連續同一 Speaker label 的片段合併為 Speaker block。

        Args:
            segments: 含 speaker／start／end／text 的 WhisperX 片段。

        Returns:
            合併後的 Speaker block 清單。
        """
        blocks: list[dict[str, Any]] = []
        block_speaker = None
        block_start = None
        block_end = None
        block_texts: list[str] = []

        for segment in segments:
            text: str = segment.get("text", "")
            text = text.strip()
            if not text:
                continue

            speaker_key = segment.get("speaker") or "UNKNOWN"
            start = segment.get("start", 0.0)
            end = segment.get("end", 0.0)

            if speaker_key != block_speaker:
                if block_speaker is not None and block_texts:
                    blocks.append(
                        {
                            "speaker": block_speaker,
                            "start": block_start,
                            "end": block_end,
                            "text": " ".join(block_texts),
                        }
                    )
                block_speaker = speaker_key
                block_start = start
                block_texts = []

            block_end = end
            block_texts.append(text)

        if block_speaker is not None and block_texts:
            blocks.append(
                {
                    "speaker": block_speaker,
                    "start": block_start,
                    "end": block_end,
                    "text": " ".join(block_texts),
                }
            )
        return blocks

    def diarize_chunk(
        self,
        flow_id: str,
        segments: Iterable[Segment],
        info: TranscriptionInfo,
        audio_path: str,
    ) -> ChunkDiarization:
        """對本包音做 WhisperX 對齊與 diarize，回傳 turns 與每位講者一條 embedding。

        不收 min／max speaker，人數交給 WhisperX 預設。內部可用詞級時間切字，但不把時間戳放進結果。

        Args:
            flow_id: 流程唯一識別符。
            segments: 已過濾的 faster-whisper 片段。
            info: 轉錄資訊（含偵測語言，供 alignment 選模）。
            audio_path: 本包 wav 路徑。

        Returns:
            時間序 turns，以及 turns 中每位講者恰好一列 speakers。

        Raises:
            CudaException: CUDA 相關失敗。
            Exception: 對齊、diarize 或 embedding 無法編成合法結果。
        """
        input_segments = self._build_transcript_segments(segments)
        if not input_segments:
            return ChunkDiarization(turns=[], speakers=[])

        aligned_result = None
        diarize_segments = None
        speaker_result = None
        embeddings = None
        try:
            lh.handle_info(
                self.logger,
                flow_id,
                get_call_site_info(),
                f"Starting Alignment. flow_id={flow_id}, audio_path={audio_path}",
            )
            alignment_model, alignment_metadata = self._load_alignment_model(
                flow_id=flow_id, language_code=info.language
            )
            aligned_result = whisperx.align(
                input_segments,
                alignment_model,
                alignment_metadata,
                audio_path,
                self.device,
            )

            lh.handle_info(
                self.logger,
                flow_id,
                get_call_site_info(),
                f"Starting Diarization. flow_id={flow_id}, audio_path={audio_path}",
            )
            diarize_segments, embeddings = self.diarization_pipeline(
                audio_path, return_embeddings=True
            )
            speaker_result = whisperx.assign_word_speakers(
                diarize_segments,
                aligned_result,
                speaker_embeddings=embeddings,
                fill_nearest=True,
            )
            return format_chunk_diarization(
                speaker_result.get("segments", []),
                embeddings,
            )
        except Exception as e:
            classified = cuda_exception_classifier(e)
            if isinstance(classified, CudaException):
                self.logger.error(
                    f"CUDA exception occurred during chunk diarization. flow_id={flow_id}, audio_path={audio_path}, error={str(e)}"
                )
                raise classified from e
            self.logger.error(
                f"Failed to diarize chunk. flow_id={flow_id}, audio_path={audio_path}, error={str(e)}"
            )
            raise
        finally:
            del aligned_result, diarize_segments, speaker_result, embeddings
            self._clear_cuda_cache()

    def _clear_cuda_cache(self) -> None:
        """
        回收未使用的 Python 物件，並釋放 PyTorch CUDA caching allocator 中的閒置快取。
        """
        gc.collect()
        if getattr(self, "device", None) == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _build_diarization_options(self, diarization_params: DiarizationParams) -> dict[str, int]:
        """
        將 API speaker 參數轉成 WhisperX diarization pipeline 可接受的選項。
        """

        diarization_options: dict[str, int] = {}
        if diarization_params.min_speaker is not None:
            diarization_options["min_speakers"] = diarization_params.min_speaker
        if diarization_params.max_speaker is not None:
            diarization_options["max_speakers"] = diarization_params.max_speaker
        return diarization_options

    def _create_profile_cache(self, alignment_profiles: list[AlignmentProfile]) -> None:
        """
        Create a cache of profile by support languages.
        """
        for profile in alignment_profiles:
            alignment_model = AlignmentInstance(profile=profile)
            for language_code in profile.get_support_languages():
                normalized_language_code = language_code.lower()
                if normalized_language_code in self.alignment_models:
                    raise ValueError(f"Duplicate alignment language code: {normalized_language_code}")
                self.alignment_models[normalized_language_code] = alignment_model
        if not self.alignment_models:
            raise ValueError("Alignment profiles must include at least one language code.")
    
    def _load_alignment_model(self, flow_id: str, language_code: str) -> tuple[Any, dict[str, Any]]:
        """
        依 language_code 載入並快取 WhisperX alignment model。
        """
        language_code = language_code.lower()
        alignment_model = self.alignment_models.get(language_code)
        if alignment_model is None:
            raise lh.handle_error(self.logger, flow_id, get_call_site_info(), f"Diarization does not supported language code: {language_code}", errorCode="MSS-VTT-000100", status_code=400)

        # return cached model and metadata
        if alignment_model.model is not None and alignment_model.metadata is not None:
            return alignment_model.model, alignment_model.metadata

        # lazy loading alignment model and metadata
        alignment_profile = alignment_model.profile
        load_language_code = alignment_profile.get_load_language_code()
        loaded_model, loaded_metadata = whisperx.load_align_model(
            model_name=alignment_profile.model_dir,
            language_code=load_language_code.lower(),
            device=self.device,
            model_cache_only=True,
        )
        # cache loaded model and metadata
        alignment_model.model = loaded_model
        alignment_model.metadata = loaded_metadata
        return alignment_model.model, alignment_model.metadata

    def _build_transcript_segments(self, segments: Iterable[Segment]) -> list[dict[str, Any]]:
        """
        將 faster-whisper Segment 轉成 WhisperX alignment 可接受的資料結構。
        """
        input_segments = []
        for seg in segments:
            segment_text = seg.text.strip()
            if not segment_text:
                continue
            input_segments.append({
                "text": segment_text,
                "start": seg.start,
                "end": seg.end,
                "avg_logprob": seg.avg_logprob,
            })
        return input_segments

    def format_transcription(self, segments: Iterable[dict[str, Any]], word_timestamps: bool = False) -> str:
        """
        將含 speaker 欄位的 WhisperX segments 合併成包含 speaker 編號的文字。
        """
        lines = []

        if not word_timestamps:
            for block in self.merge_speaker_blocks(segments):
                lines.append(
                    f"[{block['speaker']}]|{block['start']:.2f}s -> {block['end']:.2f}s| {block['text']}\n"
                )
        else:
            for segment in segments:
                speaker_key = segment.get("speaker", "UNKNOWN")                
                if not segment.get("words"):
                    continue

                for word in segment["words"]:
                    start = word.get("start", "N/A")
                    end = word.get("end", "N/A")
                    word_text = word["word"]
                    word_speaker = word.get("speaker", speaker_key) 
                    
                    if start != "N/A":
                        lines.append(f"[{word_speaker}]|{start:.2f}s -> {end:.2f}s| {word_text}\n")
        return "".join(lines).strip()
