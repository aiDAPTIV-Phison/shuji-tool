import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

class Logger:
    def __init__(self, name: str, log_file: str = "logs/app.log", max_bytes: int = 5 * 1024 * 1024, backup_count: int = 3):
        """
        初始化 Logger
        :param name: 日誌紀錄器的名稱
        :param log_file: 日誌文件的輸出路徑
        :param max_bytes: 單個日誌文件的最大大小（預設 5MB）
        :param backup_count: 日誌文件的備份數量（預設 3 個）
        """
        self.logger = logging.getLogger(name)
        level_name = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)
        self.logger.setLevel(level)
        # getLogger 回傳同名單例；重複 addHandler 會讓同一則 log 印多次
        if self.logger.handlers:
            return

        # 往上傳遞給 root，讓 OTEL LoggingHandler 能收到同一則 record
        self.logger.propagate = True

        # 檢查目錄是否存在，不存在則建立
        log_file_path = Path(log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        # --- 文件處理器（輸出到文件） ---
        file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
        file_handler.setLevel(level)

        # --- 終端處理器（輸出到命令行） ---
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)

        # --- 格式化（統一日誌格式） ---
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)   # 文件日誌格式
        stream_handler.setFormatter(formatter)  # 終端日誌格式

        # --- 添加處理器至 logger ---
        self.logger.addHandler(file_handler)    # 添加到文件
        self.logger.addHandler(stream_handler)  # 添加到終端

    def debug(self, message: str) -> None:
        """記錄 DEBUG 級別的日誌"""
        self.logger.debug(message)

    def info(self, message: str) -> None:
        """記錄 INFO 級別的日誌"""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """記錄 WARNING 級別的日誌"""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """記錄 ERROR 級別的日誌"""
        self.logger.error(message)

    def critical(self, message: str) -> None:
        """記錄 CRITICAL 級別的日誌"""
        self.logger.critical(message)
