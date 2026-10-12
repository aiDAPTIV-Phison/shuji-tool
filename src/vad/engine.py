from __future__ import annotations

import asyncio
from typing import Any

from silero_vad import get_speech_timestamps

from vad.parameters import VadParams

_SILERO_SAMPLE_RATE = 16000


def _load_wav_for_silero(audio_path: str):
    """以 TorchCodec 讀 wav，轉成 Silero 需要的 16 kHz 單聲道波形。

    Args:
        audio_path: wav 檔路徑。

    Returns:
        形狀為 ``(num_samples,)`` 的 float tensor。
    """
    # silero-vad 的 read_audio 在 torchaudio 2.8 沒有 I/O backend；
    # 改走 image／環境已依賴的 TorchCodec（ffmpeg）。
    from torchcodec.decoders import AudioDecoder

    samples = AudioDecoder(
        audio_path,
        sample_rate=_SILERO_SAMPLE_RATE,
        num_channels=1,
    ).get_all_samples()
    return samples.data.squeeze(0)


class SileroVadEngine:
    """以 silero-vad（CPU ONNX）對單一音訊窗產出 Speech timestamps。"""

    def __init__(self, model: Any):
        """保存已載入的 Silero 模型。

        Args:
            model: `load_silero_vad(onnx=True)` 的回傳值。
        """
        self._model = model

    def _build_silero_kwargs(self, params: VadParams) -> dict:
        """只轉送有值的旋鈕，對應 silero-vad 參數名。

        Args:
            params: HTTP 組出的選填旋鈕。

        Returns:
            傳給 get_speech_timestamps 的 kwargs（不含 wav／model）。
        """
        kwargs: dict = {"return_seconds": True}
        if params.vad_threshold is not None:
            kwargs["threshold"] = params.vad_threshold
        if params.vad_filter_ms is not None:
            kwargs["min_silence_duration_ms"] = params.vad_filter_ms
        if params.neg_threshold is not None:
            kwargs["neg_threshold"] = params.neg_threshold
        if params.min_speech_duration_ms is not None:
            kwargs["min_speech_duration_ms"] = params.min_speech_duration_ms
        if params.max_speech_duration_s is not None:
            kwargs["max_speech_duration_s"] = params.max_speech_duration_s
        if params.speech_pad_ms is not None:
            kwargs["speech_pad_ms"] = params.speech_pad_ms
        return kwargs

    def _detect_sync(self, audio_path: str, params: VadParams) -> list[dict[str, int]]:
        """同步讀取 wav 並呼叫 Silero。

        Args:
            audio_path: 暫存 wav 路徑。
            params: 選填偵測旋鈕。

        Returns:
            相對檔頭、單位毫秒的 Speech timestamps。
        """
        wav = _load_wav_for_silero(audio_path)
        raw = get_speech_timestamps(wav, self._model, **self._build_silero_kwargs(params))
        return [
            {"start_ms": int(round(item["start"] * 1000)), "end_ms": int(round(item["end"] * 1000))}
            for item in raw
        ]

    async def detect_speech_timestamps(
        self, flow_id: str, audio_path: str, params: VadParams
    ) -> list[dict[str, int]]:
        """對單一音訊窗偵測語音時段（執行緒池，不阻塞事件迴圈）。

        Args:
            flow_id: 流程識別（保留給 log／追蹤，本方法不使用）。
            audio_path: 暫存 wav 路徑。
            params: 選填偵測旋鈕。

        Returns:
            Speech timestamps 列表；無語音時為空列表。
        """
        del flow_id
        return await asyncio.to_thread(self._detect_sync, audio_path, params)
