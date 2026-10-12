import logging

from fastapi import HTTPException

from utils.logger import Logger


def _resolve_flow_id(flow_id):
    """將 None flow_id 正規化為可 grep 的占位字串。"""
    return flow_id if flow_id is not None else "-"


def _resolve_call_site(call_site):
    """將 None call site 正規化為可 grep 的占位字串。"""
    return call_site if call_site is not None else "-"


def _build_record_extra(flow_id, call_site, error_code=None):
    """組合寫入 LogRecord 的結構化欄位。"""
    extra = {
        "flow_id": str(_resolve_flow_id(flow_id)),
        "call_site": str(_resolve_call_site(call_site)),
    }
    if error_code is not None:
        extra["error_code"] = str(error_code)
    return extra


def _format_message(flow_id, message, error_code=None):
    """組合寫入 stdout／模組檔的訊息格式。"""
    prefix = f"[flow_id={_resolve_flow_id(flow_id)}]"
    if error_code is not None:
        return f"{prefix} [ErrorCode: {error_code}] {message}"
    return f"{prefix} {message}"


def _emit(logger: Logger, level: int, flow_id, call_site, message, error_code=None):
    """寫入傳入 logger，並在 record 上帶 flow_id 與 call site。"""
    text = _format_message(flow_id, str(message), error_code)
    logger.logger.log(
        level,
        text,
        extra=_build_record_extra(flow_id, call_site, error_code),
    )


class LogHandler:
    @staticmethod
    def handle_debug(logger: Logger, flow_id, call_site, message):
        _emit(logger, logging.DEBUG, flow_id, call_site, message)

    @staticmethod
    def handle_info(logger: Logger, flow_id, call_site, message):
        _emit(logger, logging.INFO, flow_id, call_site, message)

    @staticmethod
    def handle_warning(logger: Logger, flow_id, call_site, message):
        _emit(logger, logging.WARNING, flow_id, call_site, message)

    @staticmethod
    def handle_error(logger: Logger, flow_id, call_site, message, errorCode=None, status_code=500):
        _emit(logger, logging.ERROR, flow_id, call_site, message, errorCode)
        return HTTPException(
            status_code=status_code,
            detail={"message": str(message), "errorCode": errorCode},
        )
