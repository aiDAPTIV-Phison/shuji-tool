import os

class ConfigHandler:
    @staticmethod 
    def get_boolean(env: str, default: bool = False) -> bool:
        raw = os.environ.get(env)
        if raw is None:
            return default
        normalized = raw.strip().lower()
        if normalized in {"1", "true", "yes", "on", "enable", "enabled", "y", "t"}:
            return True
        if normalized in {"0", "false", "no", "off", "disable", "disabled", "n", "f"}:
            return False
        return default

    @staticmethod
    def get_whisper_model_folder():
        return os.environ.get("WHISPER_MODEL_FOLDER", "/model/whisper/")

    @staticmethod
    def get_alignment_model_folder():
        return os.environ.get("ALIGNMENT_MODEL_FOLDER", "/model/alignment/")

    @staticmethod
    def get_diarization_model_folder():
        return os.environ.get("DIARIZATION_MODEL_FOLDER", "/model/diarization/")

    @staticmethod
    def is_diarization_enabled() -> bool:
        return ConfigHandler.get_boolean("ENABLE_DIARIZATION", False)

    @staticmethod
    def get_app_port():
        return int(os.environ.get("APP_PORT", 7766))
    
    
    @staticmethod
    def get_app_workers():
        return int(os.environ.get("APP_WORKERS", 1))

    @staticmethod
    def get_uvicorn_root_path():
        """取得反向代理對外路徑前綴（供 Uvicorn root_path 使用）。

        Returns:
            路徑前綴字串；未設定時為空字串。
        """
        return os.environ.get("UVICORN_ROOT_PATH", "")

    @staticmethod
    def get_batch_size():
        return int(os.environ.get("WHISPER_BATCH_SIZE", 8))

    @staticmethod
    def get_whisper_beam_size() -> int:
        """取得 Whisper 解碼 beam size。

        Returns:
            環境變數 WHISPER_BEAM_SIZE；未設定時為 5。
        """
        return int(os.environ.get("WHISPER_BEAM_SIZE", 5))
    
    @staticmethod
    def get_remove_first_segment_no_speech_prob():
        return float(os.environ.get("REMOVE_FIRST_SEGMENT_NO_SPEECH_PROB", 0.55))
    
    @staticmethod
    def get_remove_first_segment_log_prob():
        return float(os.environ.get("REMOVE_FIRST_SEGMENT_LOG_PROB", -1.2))
    
    @staticmethod
    def get_remove_last_segment_duration():
        return float(os.environ.get("REMOVE_LAST_SEGMENT_DURATION", 2.5))
    
    @staticmethod
    def get_remove_last_segment_no_speech_prob():
        return float(os.environ.get("REMOVE_LAST_SEGMENT_NO_SPEECH_PROB", 0.8))
    
    @staticmethod
    def get_hard_no_speech_prob_threshold():
        return float(os.environ.get("HARD_NO_SPEECH_PROB_THRESHOLD", 0.99))